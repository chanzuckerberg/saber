from __future__ import annotations

"""
SABER Label Converter - Convert JSON annotations to SABER zarr format
"""
from typing import Dict, List, Optional, Set
from saber import cli_context
from pathlib import Path
from tqdm import tqdm
import click

class SABERLabelConverter:

    def __init__(self):
        self.label_to_index = {}
        self.discovered_labels = set()
    
    def discover_labels(self, json_path: Path) -> Set[str]:
        import json
        """
        Discover all unique labels from JSON file.
        """
        labels = set()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for frame_data in data.values():
            if frame_data:  # Skip empty frames
                labels.update(frame_data.values())
        
        return labels
    def create_label_mapping(self, 
                       labels: Set[str], 
                       custom_order: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Create label to index mapping.
        
        Args:
            labels: Set of discovered labels
            custom_order: Optional list specifying custom order (if provided, only these labels are included)
        """

        mapping = {'background': 0}  # Background is always 0
        
        if custom_order:
            # Only include labels that are in the custom order
            for i, label in enumerate(custom_order, start=1):
                if label != 'background':  # Skip background if in custom list
                    mapping[label] = i
            
            # Report which labels are being excluded
            excluded = labels - set(mapping.keys())
            if excluded:
                click.echo(f"ℹ️  Info: Excluding labels not in custom order: {sorted(excluded)}")
        else:
            # Default to alphabetical order (include all discovered labels)
            sorted_labels = sorted(labels - {'background'})
            for i, label in enumerate(sorted_labels, start=1):
                mapping[label] = i
        
        return mapping
    
    def extract_mask_values(self, masks_3d: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Extract individual masks and their associated values.
        
        For 3D mask arrays, tries to determine the unique value for each mask.
        Returns a dict mapping mask_value -> mask_array
        """
        import numpy as np

        mask_dict = {}
        
        for i, mask in enumerate(masks_3d):
            # Get unique non-zero values in this mask
            unique_vals = np.unique(mask[mask > 0])
            
            if len(unique_vals) > 0:
                # Use the first (or most common) non-zero value as the mask's ID
                mask_value = float(unique_vals[0])
            else:
                # If mask is binary or empty, use index+1 as value
                mask_value = float(i + 1)
            
            mask_dict[mask_value] = mask
        
        return mask_dict
    
    def convert(self, 
               sam2_zarr_path: Path,
               json_path: Path,
               output_path: Path,
               label_mapping: Dict[str, int]):
        """
        Convert JSON annotations to labeled zarr format compatible with SABER dataloader.
        """
        from saber.utils.zarr_writer import add_attributes
        import numpy as np
        import json, zarr

        # Load JSON annotations
        with open(json_path, 'r') as f:
            frame_labels = json.load(f)
        labels = list(label_mapping.keys())            
        
        # Load SAM2 masks
        sam2_data = zarr.open(sam2_zarr_path, 'r')
        
        # Create output zarr
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)

        # Copy over AMG metadata if present
        if 'amg' in sam2_data.attrs:
            root.attrs['amg'] = dict(sam2_data.attrs['amg'])
        
        # Process each run_id
        for run_id in tqdm(frame_labels.keys()):
            annotations = frame_labels[run_id]
            
            # Skip empty annotations
            if len(annotations) == 0:
                continue
            
            # Check if run_id exists in SAM2 data
            if run_id not in sam2_data:
                click.echo(f"⚠️  Warning: Run ID '{run_id}' not found in SAM2 zarr. Skipping.", err=True)
                continue

            # Get image data
            if '0' in sam2_data[run_id]:
                image = sam2_data[run_id]['0'][:]
            else:
                click.echo(f"⚠️  Warning: No image found for run ID '{run_id}'. Skipping.", err=True)
                continue
            
            # Get SAM2 masks
            if 'labels' in sam2_data[run_id] and '0' in sam2_data[run_id]['labels']:
                sam2_masks = sam2_data[run_id]['labels']['0'][:]
            else:
                click.echo(f"⚠️  Warning: No masks found for run ID '{run_id}'. Skipping.", err=True)
                continue
            
            im = sam2_data[run_id]['0'][:]
            (nx, ny) = im.shape
            masks0 = sam2_data[run_id]['labels']['0'][:]
            nMasks = masks0.shape[0]
            masks = np.zeros((len(labels), nx, ny), dtype=np.uint8)
            used_mask_values = set()

            # Extract mask values and create mapping
            mask_value_to_array = self.extract_mask_values(masks0)

            for seg_value, label in annotations.items():
                # Skip labels that were excluded from the mapping
                if label not in label_mapping:
                    continue
                # Add mask to the corresponding class
                mask_array = mask_value_to_array[float(seg_value)]
                masks[label_mapping[label],] = np.logical_or(
                        masks[label_mapping[label],], 
                        mask_array
                    ).astype(np.uint8)
                used_mask_values.add(float(seg_value))

            # Get rejected masks (masks not assigned to any class)
            rejected_masks = []
            for mask_value, mask_array in mask_value_to_array.items():
                if mask_value not in used_mask_values:
                    rejected_masks.append(mask_array)
            
            # Create group for this run_id - Save image
            group = root.create_group(run_id)
            group.create_dataset(
                '0', 
                data=image, 
                dtype=image.dtype, 
                compressor=zarr.Blosc(cname='zstd', clevel=2, shuffle=2)
            )
            pixel_size = sam2_data[run_id].attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][0]
            add_attributes(group, pixel_size, False)
            
            # Save class masks
            if masks.sum() > 0:
                labels_group = group.create_group('labels')
                labels_group.create_dataset(
                    '0',  # Changed from 'labels' to 'masks' to match dataloader
                    data=masks,
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname='zstd', clevel=2, shuffle=2)
                )
                add_attributes(labels_group, pixel_size, True)
            
            # Save rejected masks
            if len(rejected_masks) > 0:
                labels_group.create_dataset(
                    'rejected',
                    data=rejected_masks,
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname='zstd', clevel=2, shuffle=2)
                )
            else:
                # Empty array if no rejected masks
                labels_group.create_dataset(
                    'rejected',
                    data=np.array([]),
                    dtype=np.uint8
                )
        
        # Save metadata at root level
        root.attrs['labels'] = label_mapping
    
    def print_label_summary(self, label_mapping: Dict[str, int]):
        """
        Print a summary of the label mapping.
        """
        click.echo("\n" + "="*50)
        click.echo("LABEL MAPPING SUMMARY")
        click.echo("="*50)
        click.echo(f"Total classes: {len(label_mapping)} (including background)")
        click.echo("\nIndex → Label:")
        for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            click.echo(f"  {idx:3d} → {label}")
        click.echo("="*50)


@click.command(context_settings=cli_context)
@click.option('--input', '-i', required=True, type=click.Path(exists=True, path_type=Path),
              help='SAM2 zarr file with masks')
@click.option('--labels', '-l', default='labels.json', type=click.Path(exists=True, path_type=Path),
              help='JSON file with frame annotations (default: labels.json)')
@click.option('--output', '-o', default='labeled.zarr', type=click.Path(path_type=Path),
              help='Output zarr file (default: labeled.zarr)')
@click.option('--classes', '-c', type=str,
              help='Comma-separated label order (e.g., lysosomes,npc,edge). If not provided, uses alphabetical order.')
def labeler(input, labels, output, classes):
    """
    Apply JSON annotations to SAM2 masks for SABER training.
    
    \b
    Examples:
        # Use alphabetical ordering (default)
        saber_label_converter.py -i sam2_masks.zarr
        
        # Specify custom class order
        saber_label_converter.py -i sam2_masks.zarr --classes lysosomes,npc,edge
    """
    
    run_labeler(input, labels, output, classes)

def run_labeler(input, labels, output, classes):
    """
    Run the labeler.
    """
    import sys

    # Initialize converter
    converter = SABERLabelConverter()
    
    # Discover all labels
    click.echo(f"📂 Reading labels from: {labels}")
    all_labels = converter.discover_labels(labels)
    click.echo(f"🔍 Found {len(all_labels)} unique labels: {sorted(all_labels)}")
    
    # Parse custom class order if provided
    custom_order = None
    if classes:
        custom_order = [c.strip() for c in classes.split(',')]
        click.echo(f"📝 Using custom class order: {custom_order}")
    else:
        click.echo("📝 Using alphabetical order (no --classes specified)")
    
    # Create label mapping
    label_mapping = converter.create_label_mapping(all_labels, custom_order)
    
    # Print summary
    converter.print_label_summary(label_mapping)
    
    # Convert
    click.echo(f"\n🔄 Converting {input.name}...")
    try:
        converter.convert(
            input,
            labels,
            output,
            label_mapping
        )
        click.echo(f"✅ Successfully converted!")
        click.echo(f"   • Output saved to: {output}")
        click.echo(f"\n💡 Use --num-classes {len(label_mapping)} when training with SABER")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    labeler()