from saber.filters.downsample import FourierRescale2D
from saber.visualization.results import export_movie
from saber.segmenters.fib import fibSegmenter
from saber.classifier.models import common
from saber.utils import slurm_submit
from skimage import io as sio
import glob, click
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

def fib_options(func):
    """Decorator to add shared options for fib commands."""
    options = [
        click.option("--input", type=str, required=True,
                      help="Path to Fib or Project, in the case of project provide the file extention (e.g. 'path/*.mrc')"),
        click.option("--output", type=str, required=False, default='masks.npy',
                      help="Path to Output Segmentation Masks"),
        click.option("--ini_depth", type=int, required=False, default=10,
                      help="Spacing between slices to Segment"),
        click.option("--nframes", type=int, required=False, default=None,
                      help="Number of frames to propagate in video segmentation"),
        click.option('--scale-factor', type=float, required=False, default=1,
                      help='Scale Factor to Downsample Images. If not provided, no downsampling will be performed.'),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve order in CLI
        func = option(func)
    return func


@cli.command(context_settings={"show_default": True})
@fib_options
@slurm_submit.sam2_inputs
@slurm_submit.classifier_inputs
def fib(
    input: str,
    output: str,
    ini_depth: int,
    nframes: int,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    scale_factor: float,
    ):
    """
    Segment a Fib Volume
    """

    print(f'\nStarting Fib Segmentation for the following input: {input}')
    print(f'Segmentations will be performed every {ini_depth} slices for ±{nframes} frames')
    print(f'Output Masks will be saved to: {output}')

    # Read the Fib Volume
    volume = read_fib_volume(input, scale_factor)

    # Load the Classifier Model
    predictor = common.get_predictor(model_weights, model_config)

    # Create an instance of fibSegmenter
    segmenter = fibSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,
        target_class=target_class,
    )

    # Segment the Volume
    masks = segmenter.segment(volume, ini_depth, nframes)

    # Export the Masks as a Movie
    export_movie(volume, masks,'segmentation.gif')

    # (TODO): Save the Masks
    np.save(output, masks)

def read_fib_volume(input: str, scale_factor: float):
    """
    Read the Fib Volume from a directory or a single file
    """

    # Read the Volume from a directory or a single file
    if '*' in input:
        files = glob.glob(input)
        if len(files) == 0:
            raise ValueError(f"No files found for pattern: {input}")
        files.sort()  # Ensure files are in order
        for ii in range(len(files)):
            im = sio.imread(files[ii])
            if ii == 0:
                volume = np.zeros((len(files), im.shape[0], im.shape[1]))
            volume[ii, :, :] = im
    else:
        volume = sio.imread(input)
    volume = volume.astype(np.float32) # Convert to float32

    # Downsample if needed
    if scale_factor > 1:
        for i in range(volume.shape[0]):
            volume[i, :, :] = FourierRescale2D.run(volume[i, :, :], scale_factor)
    
    return volume

    