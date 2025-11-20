from saber.utils import slurm_submit
from saber import cli_context
import rich_click as click

def light_options(func):
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
@light_options
@slurm_submit.sam2_inputs
@slurm_submit.classifier_inputs
def light(
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
    Segment features from light microscopy movies (e.g. cells under an optical microscope).
    """ 

    run_light_segment(
        input, output, ini_depth, nframes, 
        sam2_cfg, model_weights, model_config, 
        target_class, scale_factor
    )


def run_light_segment(
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
    Segment a Light Movie
    """
    from saber.visualization.results import export_movie
    from saber.segmenters.fib import propagationSegmenter
    from saber.classifier.models import common
    import numpy as np

    print(f'\nStarting Light Movie Segmentation for the following input: {input}')
    print(f'Segmentations will be performed every {ini_depth} slices for ±{nframes} frames')
    print(f'Output Masks will be saved to: {output}')

    # Read the Fib Volume
    volume = read_light_movie(input, scale_factor)

    # Load the Classifier Model
    predictor = common.get_predictor(model_weights, model_config)

    # Create an instance of fibSegmenter
    segmenter = propagationSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,
        target_class=target_class,
        em_modality = False,
    )

    # Segment the Volume
    masks = segmenter.segment(volume, ini_depth, nframes)

    # (TODO): Save the Masks
    np.save(output, masks)

    # Export the Masks as a Movie
    export_movie(volume, masks,'segmentation.gif')

def read_light_movie(input: str, scale_factor: float):
    """
    Read the Light Movie from a directory or a single file
    """
    from saber.filters.downsample import FourierRescale2D
    import skimage.io as sio
    import numpy as np
    import glob

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
        tmp_im = FourierRescale2D.run(volume[0, :, :], scale_factor)
        out_shape = (volume.shape[0], tmp_im.shape[0], tmp_im.shape[1])
        vol_out = np.zeros(out_shape, dtype=volume.dtype)
        vol_out[0, :, :] = tmp_im
        for i in range(1, volume.shape[0]):
            vol_out[i, :, :] = FourierRescale2D.run(volume[i, :, :], scale_factor)
        volume = vol_out
    
    return volume