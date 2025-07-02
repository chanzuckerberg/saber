from saber.entry_points.inference_core import segment_micrograph_core
from saber.utils import parallelization, slurm_submit, io
from saber.segmenters.loaders import base_microsegmenter
from saber.visualization import galleries
import click, glob

@click.group()
@click.pass_context
def cli(ctx):
    pass

def micrograph_options(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--input", type=str, required=True,
                      help="Path to Micrograph or Project, in the case of project provide the file extention (e.g. 'path/*.mrc')"),
        click.option("--output", type=str, required=False, default='saber_training_data.zarr',
                      help="Path to the output Zarr file (if input points to a folder)."),
        click.option("--scale-factor", type=float, required=False, default=None, 
                      help="Scale Factor to Downsample Images. If not provided, no downsampling will be performed."),
        click.option("--target-resolution", type=float, required=False, default=None, 
                      help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed."),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

@click.command(context_settings={"show_default": True})
@micrograph_options
@slurm_submit.sam2_inputs
def prepare_micrograph_training(
    input: str, 
    output: str,
    target_resolution: float,
    scale_factor: float,
    sam2_cfg: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """    

    # Check to Make Sure Only One of the Inputs is Provided
    if target_resolution is not None and scale_factor is not None:
        raise ValueError("Please provide either target_resolution OR scale_factor input, not both.")

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

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=base_microsegmenter,
        init_args=(sam2_cfg,),
        verbose=True
    )

    # Prepare tasks
    if target_resolution is not None:
        print(f'Running SABER Segmentations with a Target Resolution of: {target_resolution} Å.')
        tasks = [ (fName, output, target_resolution, None, False, False) for fName in files ]
    elif scale_factor is not None:
        print(f'Running SABER Segmentations with a Downsampling Scale Factor of: {scale_factor}.')
        tasks = [ (fName, output, None, scale_factor, False, False) for fName in files ]
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

    print('Preparation of Saber Training Data Complete!')     
