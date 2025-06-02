from saber.process.downsample import FourierRescale2D
from saber.microSABER import cryoMicroSegmenter
from saber import io, utilities as utils
from saber.process import slurm_submit
import click, torch, zarr, os, mrcfile
from multiprocessing import Lock
import multiprocess as mp
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt


@click.group()
@click.pass_context
def cli(ctx):
    pass

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
    
    # Convert Masks to Numpy Array (Sorted by Area in Ascending Order)
    (nx, ny) = masks_list[0]['segmentation'].shape
    masks = np.zeros([len(masks_list), nx, ny], dtype=np.uint8)
    masks_list = sorted(masks_list, key=lambda mask: mask['area'], reverse=False)

    # Populate the numpy array
    for j, mask in enumerate(masks_list):
        masks[j] = mask['segmentation'].astype(np.uint8) * (j + 1)
    
    # Return the Segmented Image and Masks
    image_seg = segmenter.image
    return image_seg, masks

def extract_sam2_candidates(
    fName: str,
    fPath: str,
    pixel_size: float,
    model_cfg: str,
    zroot: zarr.Group,
    deviceID: int = 0,
    ):

    # Fourier Crop the Image to the Desired Resolution
    # image = fourier_crop_mrc_to_resolution(fPath, pixel_size)
    image = mrc_fourier_crop(fPath, pixel_size)

    # Initialize the Segmenter
    segmenter = cryoMicroSegmenter(
        sam2_cfg = model_cfg,
        deviceID = deviceID
    )    
    
    # Process Multiple Slabs or Single Slab at the Center of the Volume
    image_seg, masks = segment(segmenter, image)
    save_to_zarr(zroot, fName, image_seg, masks)

@click.command(context_settings={"show_default": True})
@click.option('--mrc-path', type=str, required=True, 
              help="Path to the input MRC files.")
@click.option('--pixel-size', type=float, required=False, default=5, 
              help="Desired Resolution to Segment Images [Angstroms].")
@click.option('--output', type=str, required=False, 
              help="Path to the output Zarr file.", 
              default = 'training_data.zarr')
@slurm_submit.sam2_inputs
def prepare_micrograph_training(
    mrc_path: str, 
    pixel_size: float,
    output: str,
    sam2_cfg: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """    

    # Set up multiprocessing - max processs = number of GPUs
    mp.set_start_method("spawn")
    n_procs = torch.cuda.device_count()    
    lock = Lock()  # Initialize the lock (Remove?)
    print(f'\nRunning SAM2 Organelle Segmentations for the Following MRC Path: {mrc_path}')
    print(f'Parallelizing the Computation over {n_procs} GPUs\n')

    # Get All MRC Files in the Directory
    fNames = os.listdir(mrc_path)
    fNames = [f for f in fNames if f.endswith('.mrc')]

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
            fPath = os.path.join(mrc_path, fName)
            print(f'\nProcessing {fPath} ({iter}/{len(fNames)})')   
            p = mp.Process(
                target=extract_sam2_candidates,
                args=(
                      fName,
                      fPath, 
                      pixel_size,
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

    print('Preparation of Cryo-SAM2 Training Data Complete!')     


@click.command(context_settings={"show_default": True})
@click.option('--mrc-path', type=str, required=True, 
              help="Path to the input MRC files.")
@click.option('--pixel-size', type=float, required=False, default=5, 
              help="Desired Resolution to Segment Images [Angstroms].")
@slurm_submit.sam2_inputs
@click.option('--output', type=str, required=True, 
              help="Path to the saved SAM2 output Zarr file.", 
              default = 'training_data.zarr')           
@slurm_submit.compute_commands
def prepare_micrograph_training_slurm(
    mrc_path: str,
    pixel_size: float,
    sam2_cfg: str,
    output: str,
    num_gpus: int,
    gpu_constraint: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """

    # Create Prepare Training Command
    command = f"""
classifier prepare-micrograph-training \\
    --mrc-path {mrc_path} \\
    --pixel-size {pixel_size} \\
    --sam2-cfg {sam2_cfg} \\
    --output {output}
    """

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="prepare-micrograph-training",
        output_file="prepare-micrograph-training.out",
        shell_name="prepare-micrograph-training.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )

# def fourier_crop_mrc_to_resolution(
def mrc_fourier_crop(
    file_path: str, 
    target_pixsize: float, 
    device=None
    ):
    """
    Read an MRC file and Fourier crop it to achieve a target pixel size (resolution)
    
    Parameters:
    -----------
    file_path : str
        Path to the MRC file
    target_pixsize : float
        Desired pixel size in Angstroms (must be larger than current_pixsize)
    device : torch.device, optional
        Device to perform computation on (defaults to cuda if available)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read MRC file and get pixel size
    with mrcfile.open(file_path) as mrc:
        current_pixsize = mrc.voxel_size.x  # in Angstroms
        image = mrc.data
    
    # Fourier Crop the Image to the Desired Resolution
    im_cropped = FourierRescale2D.run(image, current_pixsize, target_pixsize, device)

    return im_cropped