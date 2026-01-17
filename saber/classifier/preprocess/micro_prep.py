from saber.sam2.automask import amg_cli as amg
from saber.utils import slurm_submit
from saber import cli_context
import rich_click as click

@click.group()
@click.pass_context
def cli(ctx):
    pass

def micrograph_options(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("-i", "--input", type=str, required=True,
                      help="Path to Micrograph or Project, in the case of project provide the file extension (e.g. 'path/*.mrc')"),
        click.option("-o", "--output", type=str, required=False, default='training.zarr',
                      help="Path to the output Zarr file (if input points to a folder)."),
        click.option("-sf", "--scale-factor", type=float, required=False, default=None,
                      help="Scale Factor to Downsample Images. If not provided, no downsampling will be performed."),
        click.option("-tr", "--target-resolution", type=float, required=False, default=None,
                      help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed."),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

@click.command(context_settings=cli_context, name='prep2d')
@micrograph_options
@amg()
def prepare_micrograph_training(
    input: str, 
    output: str,
    target_resolution: float,
    scale_factor: float,
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
    Prepare Training Data from Micrographs for a Classifier.
    """ 

    print('⚙️  Preparing Micrograph Training Data...')
    prep2d(
        input, output, target_resolution, scale_factor, sam2_cfg,
        npoints, points_per_batch, pred_iou_thresh, crop_n_layers,
        box_nms_thresh, crop_n_points, use_m2m, multimask
    )

def prep2d(
        input, output, target_resolution, scale_factor, sam2_cfg, 
        npoints, points_per_batch, pred_iou_thresh, crop_n_layers,
        box_nms_thresh, crop_n_points, use_m2m, multimask
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """
    from saber.entry_points.inference_core import segment_micrograph_core
    from saber.segmenters.loaders import base_microsegmenter
    from saber.utils import parallelization, io
    from saber.visualization import galleries
    from saber.sam2.amg import cfgAMG    
    from skimage import io as sio
    import glob, os, shutil


    # Check to Make Sure Only One of the Inputs is Provided
    if target_resolution is not None and scale_factor is not None:
        raise ValueError("Please provide either target_resolution OR scale_factor input, not both.")

    # Prepare AMG Config
    cfg = cfgAMG(
        npoints = npoints, points_per_batch = points_per_batch, 
        pred_iou_thresh = pred_iou_thresh, box_nms_thresh = box_nms_thresh, 
        crop_n_layers = crop_n_layers, crop_n_points_downscale_factor = crop_n_points, 
        use_m2m = use_m2m, multimask_output = multimask, sam2_cfg = sam2_cfg
    )        

    # Get All Files in the Directory
    print(f'\nRunning SAM2 Training Data Preparation\nfor the Following Search Path: {input}')
    files = glob.glob(input)
    if len(files) == 0:
        raise ValueError(f"No files found in {input}")

    # Check to see if we can use target_resolution input
    if target_resolution is not None:
        image, pixel_size = io.read_micrograph(files[0])
        if pixel_size is None:
            raise ValueError(f"Pixel size is not provided for {files[0]}. Please provide scale factor input instead.")

    # Check if we need to split 3D stack into 2D slices
    image = io.read_micrograph(files[0])[0]
    if image.ndim == 3 and image.shape[0] > 3:
        files = []
        print('Writing all the slices to a temporary stack folder...')
        for ii in range(image.shape[0]):
            os.makedirs('stack', exist_ok=True)
            fname = f'stack/slice_{ii:03d}.tif'
            sio.imsave(fname, image[ii])
            files.append(fname)    

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=base_microsegmenter,
        init_args=(cfg,),
        verbose=True
    )

    # Prepare tasks
    if target_resolution is not None:
        print(f'Running SABER Segmentations with a Target Resolution of: {target_resolution} Å.')
        tasks = [ (fName, output, None, target_resolution, False, False) for fName in files ]
    elif scale_factor is not None:
        print(f'Running SABER Segmentations with a Downsampling Scale Factor of: {scale_factor}.')
        tasks = [ (fName, output, scale_factor, None, False, False) for fName in files ]
    else:  # We're not downsampling
        print('Running the Segmentations at the full micrograph resolution.')
        tasks = [ (fName, output, None, None, False, False) for fName in files ]

    # Execute
    try:
        pool.execute(
            segment_micrograph_core,
            tasks, task_ids=files,
            progress_desc="Extracting SAM2 Candidates"
        )

    finally:
        pool.shutdown()

    # Create a Gallery of the Training Data
    galleries.convert_zarr_to_gallery(output)

    # Remove the temporary stack folder if it was created
    if os.path.exists('stack'):
        shutil.rmtree('stack')

    print('Preparation of Saber Training Data Complete!')     
