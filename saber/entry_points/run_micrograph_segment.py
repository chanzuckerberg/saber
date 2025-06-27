from saber.segmenters.micro import cryoMicroSegmenter
from saber.process.downsample import FourierRescale2D
import saber.process.slurm_submit as slurm_submit
from saber.classifier.models import common
from saber.visualization import galleries 
from saber.process import mask_filters
from saber.io import read_micrograph
import glob, click, torch, zarr, os
from multiprocessing import Lock
from tqdm import tqdm
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

# Initialize the global lock at the module level
lock = Lock()

def micrograph_options(func):
    """Decorator to add shared options for micrograph commands."""
    options = [
        click.option("--input", type=str, required=True,
                      help="Path to Micrograph or Project, in the case of project provide the file extention (e.g. 'path/*.mrc')"),
        click.option("--output", type=str, required=False, default='segmentations.zarr',
                      help="Path to the output Zarr file (if input points to a folder)."),
        click.option("--target-resolution", type=float, required=False, default=None, 
              help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed."),
        click.option("--sliding-window", type=bool, required=False, default=False,
              help="Use Sliding Window for Segmentation"),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve order in CLI
        func = option(func)
    return func

# Save results to Zarr
def save_to_zarr(zroot, run_index, image, masks):

    # Convert Masks to Numpy Array if they are a list - 
    # Typical for pure SAM2 Segmentations
    if isinstance(masks, list):
        masks = mask_filters.convert_mask_list_to_array(masks)

    global lock  # Use the global lock
    with lock:
        # Create a group for the run_index
        run_group = zroot.create_group(str(run_index))

        # Save the image
        run_group.create_dataset("image", data=image, dtype="float32", overwrite=True)
        run_group.create_dataset("masks", data=masks, dtype="uint8", overwrite=True)


def segment_micrograph_separate_process(
    input: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    zroot: zarr.Group = None,
    display_image: bool = False,
    use_sliding_window: bool = True,
    target_resolution: float = None
    ):

    # Initialize the Domain Expert Classifier   
    predictor = common.get_predictor(model_weights, model_config)

    # Create an instance of cryoMicroSegmenter
    segmenter = cryoMicroSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,          # if you have a classifier; otherwise, leave as None
        target_class=target_class     # desired target class if using a classifier
    )

    # Get Micrograph
    print(f'Getting Micrograph from {input}')
    im, pixel_size = read_micrograph(input)
    im = im.astype(np.float32)

    # Downsample if desired resolution is larger than current resolution
    if target_resolution is not None and target_resolution > pixel_size:
        scale = target_resolution / pixel_size
        im = FourierRescale2D.run(im, scale)

    # Segment Micrograph
    masks = segmenter.segment_image(
        im, display_image=display_image,
        use_sliding_window=use_sliding_window,
        )

    # Only Save if the Zarr Root is Provided
    if zroot is not None:
        fName = os.path.splitext(os.path.basename(input))[0]
        save_to_zarr(zroot, fName, im, masks)

@cli.command(context_settings={"show_default": True})
@micrograph_options
@slurm_submit.sam2_inputs
@slurm_submit.classifier_inputs
def micrographs(
    input: str,
    output: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
    target_resolution: float
    ):
    """
    Segment a single micrograph or all micrographs in a folder.
    """

    files = glob.glob(input)
    if len(files) == 0:
        raise ValueError(f"No files found in {input}")
    elif len(files) == 1:
        segment_micrograph_separate_process(
            files[0], sam2_cfg, model_weights, model_config, target_class,
            display_image=True, use_sliding_window=sliding_window, 
            target_resolution=target_resolution,
            )
    else: # Processing Project, will save results to Zarr file
        import multiprocess as mp
        mp.set_start_method("spawn")

        # Initialize the shared Zarr file with the new structure
        zarr_store = zarr.DirectoryStore(output)
        zroot = zarr.group(zarr_store, overwrite=True)

        # TODO: What Attributes to Save? 

        n_procs = torch.cuda.device_count()
        print(f'\nRunning Saber Segmentations for {len(files)} Micrographs:\n')
        print(f'Paraellizing the Computation over {n_procs} GPUs\n')

        # Main Loop - Segment All Micrographs
        iter = 1
        for _iz in range(0, len(files), n_procs):
            processes = []
            for _in in range(n_procs):
                _iz_this = _iz + _in
                if _iz_this >= len(files):
                    break
                file = files[_iz_this]
                print(f'\nProcessing {file} ({iter}/{len(files)})')    
                p = mp.Process(
                    target=segment_micrograph_separate_process,
                    args=(file, 
                        sam2_cfg,
                        model_weights,
                        model_config,
                        target_class,
                        zroot,
                        False,
                        sliding_window,
                        target_resolution,
                        ),
                )
                processes.append(p)
                iter += 1

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            for p in processes:
                p.close()

        # Create a Gallery of Segmentations
        galleries.convert_zarr_to_gallery(output)

        print('Completed 2D Segmentations with Saber!')
    
@cli.command(context_settings={"show_default": True})
@micrograph_options
@slurm_submit.classifier_inputs
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def micrographs_slurm(
    input: str,
    output: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
    target_resolution: float,
    num_gpus: int,
    gpu_constraint: str
    ):
    """
    Generate a SLURM submission to segment all micrographs in a project.
    """
    
    pass

