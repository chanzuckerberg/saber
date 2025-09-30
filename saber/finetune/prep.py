"""
Convert copick data to 3D zarr format with simple JSON segmentation mapping.
Saves full 3D volumes and creates a simple ID-to-organelle mapping sorted by volume.
"""

from saber.utils.zarr_writer import add_attributes
from typing import Dict, List, Optional, Tuple
from copick_utils.io import readers
from skimage.measure import label
import copick, json, click
from tqdm import tqdm
import numpy as np
import zarr


def process_run_3d_simple(
    run,
    voxel_spacing: float,
    tomo_alg: str,
    organelle_names: List[str],
    min_component_volume: int = 100,
    user_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Process a single run in full 3D with simple volume-based sorting.

    Returns:
        volume_3d: Full 3D tomogram
        seg_3d: 3D segmentation with unique labels sorted by volume
        id_to_organelle: Dictionary mapping mask indices to organelle names
    """

    click.echo("  Loading tomogram...")

    # Get tomogram data
    vs = run.get_voxel_spacing(voxel_spacing)
    tomograms = vs.get_tomograms(tomo_alg)

    if not tomograms:
        raise ValueError(f"No tomograms found for run {run.name}")

    volume_3d = tomograms[0].numpy()
    click.echo(f"    Tomogram shape: {volume_3d.shape}")

    components = []
    offset = 1
    temp_seg_3d = np.zeros_like(volume_3d, dtype=np.uint32)

    for organelle_name in organelle_names:
        try:
            click.echo(f"    Processing {organelle_name}...")
            seg = readers.segmentation(run, voxel_spacing, organelle_name, user_id=user_id)

            if seg.shape != volume_3d.shape:
                temp_seg_3d = np.zeros_like(seg, dtype=np.uint32)

            if seg is not None:
                # Convert to binary and separate connected components
                binary_mask = (seg > 0.5).astype(np.uint8)
                labeled_mask = label(binary_mask, connectivity=3)

                # Most efficient: process in a single pass
                unique_labels, counts = np.unique(labeled_mask[labeled_mask > 0], return_counts=True)

                for label_val, vol in zip(unique_labels, counts):
                    if vol >= min_component_volume:
                        temp_seg_3d[labeled_mask == label_val] = offset
                        components.append((organelle_name, offset, vol))
                        offset += 1
            else:
                click.echo("      No segmentation found")
        except Exception as e:
            click.echo(f"      Error processing {organelle_name}: {e}")

    # Sort by volume (smallest first, so small objects are on top in GUI)
    components.sort(key=lambda x: x[2])

    # Create final segmentation with remapped labels and the mapping dictionary
    seg_3d = np.zeros_like(temp_seg_3d, dtype=np.uint16)
    id_to_organelle: Dict[str, str] = {}

    for new_label, (organelle_name, old_label, _volume) in enumerate(components, start=1):
        seg_3d[temp_seg_3d == old_label] = new_label
        id_to_organelle[str(new_label)] = organelle_name

    return volume_3d, seg_3d, id_to_organelle


def convert_copick_to_3d_zarr(
    config_path: str,
    output_zarr_path: str,
    output_json_path: Optional[str],
    voxel_spacing: float,
    tomo_alg: str,
    specific_runs: Optional[List[str]],
    min_component_volume: int,
    compress: bool,
    user_id: Optional[str],
):
    """
    Convert copick data to 3D zarr format with JSON segmentation mapping.
    """

    # Initialize copick
    root = copick.from_file(config_path)

    # Get organelle names (non-particle objects)
    organelle_names = [obj.name for obj in root.pickable_objects if not obj.is_particle]

    # Optional: filter out specific organelles
    organelle_names = [x for x in organelle_names if "membrane" not in x]
    click.echo(f"Found organelles: {organelle_names}")

    # Set default JSON output path
    if output_json_path is None:
        output_json_path = output_zarr_path.replace(".zarr", "_mapping.json")

    # Initialize zarr store
    compressor = zarr.Blosc(cname="zstd", clevel=2) if compress else None
    store = zarr.DirectoryStore(output_zarr_path)
    zroot = zarr.group(store=store, overwrite=True)

    # Master mapping dictionary
    master_mapping: Dict[str, Dict[str, str]] = {}

    # Determine which runs to process
    runs_to_process = specific_runs if specific_runs else [run.name for run in root.runs]

    for run_name in tqdm(runs_to_process, desc="Processing runs"):
        click.echo(f"\nProcessing run: {run_name}")
        run = root.get_run(run_name)

        try:
            # Process the 3D data
            volume_3d, seg_3d, id_to_organelle = process_run_3d_simple(
                run=run,
                voxel_spacing=voxel_spacing,
                tomo_alg=tomo_alg,
                organelle_names=organelle_names,
                min_component_volume=min_component_volume,
                user_id=user_id,
            )

            # Create zarr group for this run
            run_group = zroot.create_group(run_name)

            # Save 3D volume
            voxel_spacing_nm = voxel_spacing / 10  # Convert to nm
            run_group.create_dataset(
                "0",
                data=volume_3d,
                chunks=(1, volume_3d.shape[1], volume_3d.shape[2]),  # Chunk by slices
                compressor=compressor,
                dtype=volume_3d.dtype,
            )
            add_attributes(run_group, voxel_spacing_nm, True, voxel_spacing_nm)

            # Save 3D segmentation/label volume
            label_group = run_group.create_group("labels")
            label_group.create_dataset(
                "0",
                data=seg_3d,
                chunks=(1, volume_3d.shape[1], volume_3d.shape[2]),
                compressor=compressor,
                dtype=seg_3d.dtype,
            )
            add_attributes(label_group, voxel_spacing_nm, True, voxel_spacing_nm)

            # Add to master mapping
            master_mapping[run_name] = id_to_organelle

            click.echo(f"  ✅ Saved {run_name} with {len(id_to_organelle)} segmentations")

        except Exception as e:
            click.echo(f"  ❌ Error processing {run_name}: {e}")
            continue

    # Save master JSON mapping
    with open(output_json_path, "w") as f:
        json.dump(master_mapping, f, indent=2)

    click.echo("\n🎉 Conversion complete!")
    click.echo(f"📁 Zarr output: {output_zarr_path}")
    click.echo(f"📄 JSON mapping: {output_json_path}")
    click.echo(f"📊 Total runs processed: {len(master_mapping)}")


def load_3d_zarr_data(zarr_path: str, run_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load 3D data from zarr for a specific run.

    Returns:
        volume: 3D tomogram
        labels: 3D label volume
        id_to_organelle: Mapping of label values to organelle names
    """
    store = zarr.DirectoryStore(zarr_path)
    zroot = zarr.group(store=store, mode="r")

    if run_name not in zroot:
        raise ValueError(f"Run {run_name} not found in zarr")

    run_group = zroot[run_name]

    # NOTE: This mirrors the writer above: datasets saved under '0' and 'labels/0'.
    volume = run_group["0"][:]
    labels = run_group["labels"]["0"][:]
    # If you want to persist per-run mapping inside zarr attrs instead of the external JSON,
    # you can set and read it here. Currently, mappings are stored in the external JSON.
    id_to_organelle = {}

    return volume, labels, id_to_organelle


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    default="config.json",
    help="Path to copick config file.",
)
@click.option(
    "--output-zarr",
    "output_zarr_path",
    type=click.Path(dir_okay=True, writable=True, path_type=str),
    required=True,
    help="Output path for the zarr directory.",
)
@click.option(
    "--output-json",
    "output_json_path",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    help="Output path for JSON mapping (defaults to <output-zarr>_mapping.json).",
)
@click.option(
    "--voxel-size",
    "voxel_spacing",
    type=float,
    default=7.84,
    help="Voxel spacing for the tomogram data (Å).",
)
@click.option(
    "--tomo-alg",
    type=str,
    default="wbp-denoised-ctfdeconv",
    help="Tomogram algorithm to use for processing.",
)
@click.option(
    "--specific-run",
    "specific_runs",
    multiple=True,
    help="Process only specific runs. Repeat this option for multiple runs.",
)
@click.option(
    "--min-component-volume",
    type=int,
    default=10000,
    help="Minimum connected-component volume (in voxels).",
)
@click.option(
    "--user-id",
    type=str,
    default=None,
    help="UserID for accessing segmentation.",
)
@click.option(
    "--no-compress",
    is_flag=True,
    default=False,
    help="Disable compression for zarr storage.",
)
def main(
    config_path: str,
    output_zarr_path: str,
    output_json_path: Optional[str],
    voxel_spacing: float,
    tomo_alg: str,
    specific_runs: Optional[List[str]],
    min_component_volume: int,
    user_id: Optional[str],
    no_compress: bool,
):
    """Convert copick data to 3D zarr format with JSON segmentation mapping."""
    convert_copick_to_3d_zarr(
        config_path=config_path,
        output_zarr_path=output_zarr_path,
        output_json_path=output_json_path,
        voxel_spacing=voxel_spacing,
        tomo_alg=tomo_alg,
        specific_runs=list(specific_runs) if specific_runs else None,
        min_component_volume=min_component_volume,
        compress=not no_compress,
        user_id=user_id,
    )


if __name__ == "__main__":
    main()