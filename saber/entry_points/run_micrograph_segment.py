import saber.utils.slurm_submit as slurm_submit
from saber.sam2.automask import amg_cli as amg
from saber import cli_context
import rich_click as click

@click.group()
@click.pass_context
def cli(ctx):
    pass

def micrograph_options(func):
    """Decorator to add shared options for micrograph commands."""
    options = [
        click.option("-i", "--input", type=str, required=True,
                      help="Path to Micrograph or Project, in the case of project provide the file extension (e.g. 'path/*.mrc')"),
        click.option("-o", "--output", type=str, required=False, default='segmentations.zarr',
                      help="Path to the output Zarr file (if input points to a folder)."),
        click.option("-tr", "--target-resolution", type=float, required=False, default=None, 
              help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed."),
        click.option("-sf", "--scale-factor", type=float, required=False, default=None, 
              help="Scale Factor to Downsample Images. If not provided, no downsampling will be performed."),
        click.option("-sw", "--sliding-window", type=bool, required=False, default=False,
              help="Use Sliding Window for Segmentation"),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve order in CLI
        func = option(func)
    return func
    
def interactive_segment_micrograph(
    input: str, cfg,
    model_weights, model_config,
    target_class: int = -1,
    target_resolution: float = None,
    scale_factor: float = None,
    display_image: bool = False,
    use_sliding_window: bool = False,
    ):
    from saber.segmenters.micro import cryoMicroSegmenter
    from saber.filters.downsample import FourierRescale2D
    from saber.classifier.models import common
    import saber.utils.io as io
    import numpy as np
    """
    Segment a single micrograph using SABER.
    """

    # Initialize the Domain Expert Classifier   
    predictor = common.get_predictor(model_weights, model_config)

    segmenter = cryoMicroSegmenter(
            cfg=cfg,
            classifier=predictor,         # if you have a classifier; otherwise, leave as None
            target_class=target_class )   # desired target class if using a classifier

    # Let Users Save Segmentations when interactively segmenting
    if display_image:
        segmenter.save_button = True

    # Read the Micrograph
    image, pixel_size = io.read_micrograph(input)
    image = image.astype(np.float32)

    # Downsample if desired resolution is larger than current resolution
    if target_resolution is not None and target_resolution > pixel_size:
        scale = target_resolution / pixel_size
        image = FourierRescale2D.run(image, scale)
    elif scale_factor is not None:
        image = FourierRescale2D.run(image, scale_factor)

    # Produce Initialial Segmentations with SAM2
    segmenter.segment( image, display_image=True, use_sliding_window=use_sliding_window )

########################################################
# CLI Commands
########################################################

@click.command(context_settings=cli_context)
@micrograph_options
@amg()
@slurm_submit.classifier_inputs
def micrographs(
    input: str,
    output: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
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
    Segment a single micrograph or all micrographs in a folder.
    """

    print('🎨 Starting micrograph segmentation...')
    run_micrograph_segment(
        input, output, model_weights, model_config, target_class, 
        sliding_window, target_resolution, scale_factor, sam2_cfg,
        npoints, points_per_batch, pred_iou_thresh, crop_n_layers, 
        box_nms_thresh, crop_n_points, use_m2m, multimask
    )

def run_micrograph_segment(
    input: str,
    output: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
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
    Segment a single micrograph or all micrographs in a folder.
    """
    from saber.entry_points.inference_core import segment_micrograph_core
    from saber.segmenters.loaders import micrograph_workflow
    from saber.visualization import galleries 
    from saber.utils import parallelization
    from saber.sam2.amg import cfgAMG
    import glob

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
    files = glob.glob(input)
    if len(files) == 0:
        raise ValueError(f"No files found in {input}")
    elif len(files) == 1:
        print(f'Running SABER on : {files[0]}')
        # Load the Model 
        interactive_segment_micrograph(
            files[0], cfg, model_weights, model_config, target_class,
            display_image=True, use_sliding_window=sliding_window, 
            target_resolution=target_resolution, scale_factor=scale_factor
            )
        return
    print(f'\nRunning SABER Segmentations\nfor the Following Search Path: {input}')   

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=micrograph_workflow,
        init_args=(cfg, model_weights, model_config, target_class),
        verbose=True
    )

    # Prepare tasks
    if target_resolution is not None:
        print(f'Running SABER Segmentations with a Target Resolution of: {target_resolution} Å.')
        tasks = [ (fName, output, target_resolution, None, False, sliding_window) for fName in files ]
    elif scale_factor is not None:
        print(f'Running SABER Segmentations with a Downsampling Scale Factor of: {scale_factor}.')
        tasks = [ (fName, output, None, scale_factor, False, sliding_window) for fName in files ]
    else:  # We're not downsampling
        print('Running the Segmentations at the full micrograph resolution.')
        tasks = [ (fName, output, None, None, False, sliding_window) for fName in files ]

    # Execute
    try:
        pool.execute(
            segment_micrograph_core,
            tasks, task_ids=files,
            progress_desc="Running 2D-SABER"
        )

    finally:
        pool.shutdown()

    # Create a Gallery of the Training Data
    galleries.convert_zarr_to_gallery(output)
    
