from saber.utils import slurm_submit
from saber import cli_context
import rich_click as click

def fib_options(func):
    """Decorator to add shared options for fib commands."""
    options = [
        click.option("-i", "--input", type=str, required=True,
                      help="Path to Fib or Project, in the case of project provide the file extention (e.g. 'path/*.mrc')"),
        click.option("-o", "--output", type=str, required=False, default='masks.npy',
                      help="Path to Output Segmentation Masks"),
        click.option("-d", "--ini_depth", type=int, required=False, default=10,
                      help="Spacing between slices to Segment"),
        click.option("-f", "--nframes", type=int, required=False, default=None,
                      help="Number of frames to propagate in video segmentation"),
        click.option('-sf', '--scale-factor', type=float, required=False, default=1,
                      help='Scale Factor to Downsample Images. If not provided, no downsampling will be performed.'),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve order in CLI
        func = option(func)
    return func


@click.command(context_settings=cli_context)
@fib_options
@slurm_submit.classifier_inputs
def fib(
    input: str,
    output: str,
    ini_depth: int,
    nframes: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    scale_factor: float,
    ):
    """
    Segment a Fib Volume
    """

    run_fib_segment(
        input, output, ini_depth, nframes, 
        model_weights, model_config, 
        target_class, scale_factor
    )


def run_fib_segment(
    input: str,
    output: str,
    ini_depth: int,
    nframes: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    scale_factor: float,
):
    """
    Segment a Fib Volume
    """
    from saber.visualization.results import export_movie
    from saber.segmenters.propagation import propagationSegmenter
    from saber.classifier.models import common
    from saber.utils import io 
    import numpy as np

    print(f'\nStarting Fib Segmentation for the following input: {input}')
    print(f'Segmentations will be performed every {ini_depth} slices for ±{nframes} frames')
    print(f'Output Masks will be saved to: {output}')

    # Read the Fib Volume
    volume = io.read_movie(input, scale_factor)

    # Load the Classifier Model
    predictor = common.get_predictor(model_weights, model_config)

    # Create an instance of fibSegmenter
    segmenter = propagationSegmenter(
        classifier=predictor,
        target_class=target_class,
    )

    # Segment the Volume
    masks = segmenter.segment(volume, ini_depth, nframes)

    # (TODO): Save the Masks
    np.save(output, masks)

    # Export the Masks as a Movie
    export_movie(volume, masks,'segmentation.gif')

