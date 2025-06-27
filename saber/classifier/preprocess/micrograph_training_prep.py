from saber.process.downsample import FourierRescale2D
from saber.segmenters.micro import cryoMicroSegmenter
from saber import io, utilities as utils
from saber.process import slurm_submit
from saber.process import mask_filters
import click, torch, zarr, os, glob
from saber.io import read_micrograph
from multiprocessing import Lock
import multiprocess as mp
from tqdm import tqdm
import numpy as np

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
        click.option("--target-resolution", type=float, required=False, default=None, 
                      help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed."),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

# Initialize the global lock at the module level
lock = Lock()

# Save results to Zarr
def save_to_zarr(zroot, run_index, image, masks):
    global lock  # Use the global lock
    with lock:
        # Create a group for the run_index
        run_group = zroot.create_group(str(run_index))

        # Save the image
        run_group.create_dataset("image", data=image, dtype="float32", overwrite=True)
        run_group.create_dataset("masks", data=masks, dtype="uint8", overwrite=True)

# Base segmentation function that processes a given slab using the segmenter.
def segment(segmenter, image):
    
    # Produce Initialial Segmentations with SAM2
    masks_list = segmenter.segment_image(
        image, display_image=False)

    # Convert Masks to Numpy Array
    masks = mask_filters.convert_mask_list_to_array(masks_list)
    
    # Return the Segmented Image and Masks
    image_seg = segmenter.image
    return image_seg, masks

def extract_sam2_candidates(
    fName: str,
    target_resolution: float,
    model_cfg: str,
    zroot: zarr.Group,
    deviceID: int = 0,
    ):

    # Read the Micrograph
    image, pixel_size = read_micrograph(fPath)
    image = image.astype(np.float32)

    # Downsample if desired resolution is larger than current resolution
    if target_resolution is not None and target_resolution > pixel_size:
        scale = target_resolution / pixel_size
        image = FourierRescale2D.run(image, scale)

    # Initialize the Segmenter
    segmenter = cryoMicroSegmenter(
        sam2_cfg = model_cfg,
        deviceID = deviceID
    )    
    
    # Process Multiple Slabs or Single Slab at the Center of the Volume
    image_seg, masks = segment(segmenter, image)
    save_to_zarr(zroot, fName, image_seg, masks)

@click.command(context_settings={"show_default": True})
@micrograph_options
@slurm_submit.sam2_inputs
def prepare_micrograph_training(
    input: str, 
    output: str,
    target_resolution: float,
    sam2_cfg: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """    

    # Set up multiprocessing - max processs = number of GPUs
    mp.set_start_method("spawn")
    n_procs = torch.cuda.device_count()    
    lock = Lock()  # Initialize the lock (Remove?)
    print(f'\nRunning Saber Segmentations for the Following MRC Path: {mrc_path}')
    print(f'Parallelizing the Computation over {n_procs} GPUs\n')

    # Get All MRC Files in the Directory
    fNames = glob.glob(input)
    if len(fNames) == 0:
        raise ValueError(f"No files found in {input}")

    # Initialize the shared Zarr file with the new structure
    zarr_store = zarr.DirectoryStore(output)
    zroot = zarr.group(zarr_store, overwrite=True)

    iter = 1
    n_fNames = len(fNames)
    # Main Loop - Segment All Tomograms
    for _iz in range(0, n_fNames, n_procs):
        processes = []
        for _in in range(n_procs):
            _iz_this = _iz + _in
            if _iz_this >= n_fNames:
                break
            fName = fNames[_iz_this]
            print(f'\nProcessing {fName} ({iter}/{len(fNames)})')   
            p = mp.Process(
                target=extract_sam2_candidates,
                args=(
                      fName,
                      target_resolution, 
                      sam2_cfg,
                      zroot,
                      _in),
            )
            processes.append(p)
            iter += 1

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        for p in processes:
            p.close()

    print('Preparation of Saber Training Data Complete!')     


@click.command(context_settings={"show_default": True}, name='prepare-micrograph-training')
@micrograph_options
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def prepare_micrograph_training_slurm(
    input: str,
    output: str,
    target_resolution: float,
    sam2_cfg: str,
    num_gpus: int,
    gpu_constraint: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """

    # Create Prepare Training Command
    command = f"""
saber classifier prepare-micrograph-training \\
    --input {input} \\
    --output {output} \\
    --sam2-cfg {sam2_cfg} \\
    --num-gpus {num_gpus} \\
    --gpu-constraint {gpu_constraint} \\
    """

    if target_resolution is not None:
        command += f" --target-resolution {target_resolution}"

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="prepare-micrograph-training",
        output_file="prepare-micrograph-training.out",
        shell_name="prepare-micrograph-training.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )

