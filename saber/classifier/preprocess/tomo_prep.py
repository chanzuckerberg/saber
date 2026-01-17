from saber.sam2.automask import amg_cli as amg
from saber.classifier import validate_odd
from saber.utils import slurm_submit
from saber import cli_context
import rich_click as click

@click.group()
@click.pass_context
def cli(ctx):
    pass

# Base segmentation function that processes a given slab using the segmenter.
def segment(segmenter, vol, slab_thickness, zSlice):
    from saber.filters import masks as mask_filters

    # Produce Initialial Segmentations with SAM2
    segmenter.segment_slab(
        vol, slab_thickness, display_image=False, zSlice=zSlice)
    (image0, masks_list) = (segmenter.image0, segmenter.masks)
    masks_list = sorted(masks_list, key=lambda mask: mask['area'], reverse=False)
    
    # Convert Masks to Numpy Array
    masks = mask_filters.masks_to_array(masks_list)

    return image0, masks

def extract_sam2_candidates(
    run, 
    output,
    voxel_size: int, 
    tomogram_algorithm: str,
    slab_thickness: int,
    multiple_slabs: int,
    gpu_id,     # Added by GPUPool
    models      # Added by GPUPool
    ):
    from saber.utils import zarr_writer
    from copick_utils.io import readers
    import numpy as np

    # Use pre-loaded segmenter
    segmenter = models['segmenter']

    # Get the Global Zarr Writer
    zwriter = zarr_writer.get_zarr_writer(output)
    zwriter.set_dict_attr('amg', segmenter.cfg)

    # Get Tomogram
    vol = readers.tomogram(run, voxel_size, tomogram_algorithm)
    if vol is None:
        print('No Tomogram Found for Run: ', run.name)
        return
    
    # Hard coded conversion from Angstroms to nanometers
    # Copick tomograms are typically stored in Angstroms
    voxel_size /= 10
    
    # Process Multiple Slabs or Single Slab at the Center of the Volume
    if multiple_slabs > 1:
        
        # Get the Center of the Volume
        depth = vol.shape[0]
        center_index = depth // 2
        
        # Process multiple slabs centered on the volume
        for i in range(multiple_slabs):
            
            # Define the center of the slab
            offset = (i - multiple_slabs // 2) * slab_thickness
            slab_center = center_index + offset
            image_seg, masks = segment(segmenter, vol, slab_thickness, zSlice=slab_center)
            
            # Save to a group with name: run.name + "_{index}"
            group_name = f"{run.name}_{i+1}"
            zwriter.write(
                run_name=group_name, image=image_seg, 
                masks=masks.astype(np.uint8), pixel_size=voxel_size)            
    else:
        zSlice = int(vol.shape[0] // 2)
        image_seg, masks = segment(segmenter, vol, slab_thickness, zSlice=zSlice)

        # Write Run to Zarr
        zwriter.write(
            run_name=run.name, image=image_seg, 
            masks=masks.astype(np.uint8), pixel_size=voxel_size)

@click.command(context_settings=cli_context, name='prep3d')
@slurm_submit.copick_commands
@amg()
@click.option("-o", "--output", type=str, required=False, help="Path to the output Zarr file.", 
              default = 'training.zarr')
@click.option('--num-slabs', type=int, default=1, callback=validate_odd, 
              help="Number of slabs to segment per tomogram.")
def prepare_tomogram_training(
    config: str, voxel_size: int, tomo_alg: str, 
    slab_thickness: int, num_slabs: int,
    output: str,
    sam2_cfg: str,
    npoints: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    crop_n_layers: int,
    box_nms_thresh: float,
    crop_n_points: int,
    use_m2m: bool,
    multimask: bool,
    ):
    """
    Prepare Training Data from Tomograms for a Classifier.
    """

    print('⚙️  Preparing Tomogram Training Data...')
    prep3d(
        config, voxel_size, tomo_alg, slab_thickness, output, num_slabs, sam2_cfg, 
        npoints, points_per_batch, pred_iou_thresh, crop_n_layers, 
        box_nms_thresh, crop_n_points, use_m2m, multimask
    )


def prep3d(
        config, voxel_size, tomo_alg, slab_thickness, output, num_slabs, sam2_cfg, npoints, 
        points_per_batch, pred_iou_thresh, crop_n_layers, box_nms_thresh, crop_n_points, 
        use_m2m, multimask ):
    """
    Prepare Training Data from Tomograms for a Classifier.
    """ 

    from saber.segmenters.loaders import base_tomosegmenter
    from saber.visualization import galleries
    from saber.utils import parallelization
    from saber.sam2.amg import cfgAMG    
    import copick

    print(f'\nRunning SAM2 Training Data Preparation')
    print(f'Algorithm: {tomo_alg}, Voxel-Size: {voxel_size} Å')
    print(f'Using {num_slabs} slabs with {slab_thickness} A thickness')

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]
    print(f'Processing {len(run_ids)} runs for training data extraction')

    # Prepare AMG Config
    cfg = cfgAMG(
        npoints = npoints, points_per_batch = points_per_batch, 
        pred_iou_thresh = pred_iou_thresh, box_nms_thresh = box_nms_thresh, 
        crop_n_layers = crop_n_layers, crop_n_points_downscale_factor = crop_n_points, 
        use_m2m = use_m2m, multimask_output = multimask, sam2_cfg = sam2_cfg
    )

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=base_tomosegmenter,
        init_args=(cfg,),
        verbose=True
    )

    # Prepare tasks
    tasks = [
        (run, output, voxel_size, tomo_alg, slab_thickness, num_slabs)
        for run in root.runs
    ]

    # Execute
    try:
        pool.execute(
            extract_sam2_candidates,
            tasks, task_ids=run_ids,
            progress_desc="Extracting SAM2 Candidates"
        )

    finally:
        pool.shutdown()

    # Create a Gallery of the Training Data
    galleries.convert_zarr_to_gallery(output)

    print('Preparation of SABER Training Data Complete!')     
