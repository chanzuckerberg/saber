import logging
# Root logger - blocks all INFO messages
logging.getLogger().setLevel(logging.WARNING)  

from saber.tomoSABER import cryoTomoSegmenter
from saber.classifier import validate_odd
from saber import io, utilities as utils
from saber.process import slurm_submit
import copick, click, torch, zarr
from multiprocessing import Lock
import multiprocess as mp
from tqdm import tqdm
import numpy as np

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
def segment(segmenter, vol, slab_thickness, zSlice):
    
    # Produce Initialial Segmentations with SAM2
    masks_list = segmenter.segment_image(
        vol, slab_thickness, display_image=False, zSlice=zSlice)
    
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
    run, 
    voxel_size: int, 
    tomogram_algorithm: str,
    slab_thickness: int,
    model_cfg: str,
    zroot: zarr.Group,
    deviceID: int,
    multiple_slabs: int,
    ):

    # Get Tomogram
    vol = io.get_tomogram(run, voxel_size, algorithm = tomogram_algorithm)

    # Initialize the Segmenter
    segmenter = cryoTomoSegmenter(
        sam2_cfg = model_cfg,
        deviceID = deviceID
    )    
    
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
            save_to_zarr(zroot, group_name, image_seg, masks)
    else:
        zSlice = int(vol.shape[0] // 2)
        image_seg, masks = segment(segmenter, vol, slab_thickness, zSlice=zSlice)
        save_to_zarr(zroot, run.name, image_seg, masks)

@click.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.sam2_inputs
@click.option('--zarr-path', type=str, required=False, help="Path to the output Zarr file.", 
              default = '24jul29c_training_data.zarr')
@click.option('--num-slabs', type=int, default=1, callback=validate_odd, 
              help="Number of slabs to segment per tomogram.")
def prepare_tomogram_training(
    config: str, 
    voxel_size: int, 
    tomogram_algorithm: str, 
    slab_thickness: int,
    zarr_path: str,
    sam2_cfg: str,
    num_slabs: int,
    ):

    # Set up multiprocessing - max processs = number of GPUs
    mp.set_start_method("spawn")
    n_procs = torch.cuda.device_count()    
    lock = Lock()  # Initialize the lock (Remove?)
    print(f'\nRunning SAM2 Organelle Segmentations for the Following Tomograms:\n Algorithm: {tomogram_algorithm}, Voxel-Size: {voxel_size} Å')
    print(f'Paraellizing the Computation over {n_procs} GPUs\n')

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]

    # Initialize the shared Zarr file with the new structure
    zarr_store = zarr.DirectoryStore(zarr_path)
    zroot = zarr.group(zarr_store, overwrite=True)

    iter = 1
    n_run_ids = len(run_ids)
    # Main Loop - Segment All Tomograms
    for _iz in range(0, n_run_ids, n_procs):
        processes = []
        for _in in range(n_procs):
            _iz_this = _iz + _in
            if _iz_this >= n_run_ids:
                break
            run_id = run_ids[_iz_this]
            run = root.get_run(run_id)
            print(f'\nProcessing {run_id} ({iter}/{len(run_ids)})')    
            p = mp.Process(
                target=extract_sam2_candidates,
                args=(run, 
                      voxel_size, 
                      tomogram_algorithm,
                      slab_thickness,
                      sam2_cfg,
                      zroot,
                      _in,
                      num_slabs),
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
@slurm_submit.copick_commands
@slurm_submit.sam2_inputs
@click.option('--zarr-path', type=str, required=True, help="Path to the saved SAM2 output Zarr file.", 
              default = '24jul29c_training_data.zarr')
@click.option('--num-slabs', type=int, default=1, callback=validate_odd, 
              help="Number of slabs to segment per tomogram.")              
@slurm_submit.compute_commands
def prepare_tomogram_training_slurm(
    config: str,
    sam2_cfg: str,
    voxel_size: int, 
    tomogram_algorithm: str,
    slab_thickness: int,
    zarr_path: str,
    num_gpus: int,
    gpu_constraint: str,
    num_slabs: int,
    ):

    # Create Prepare Training Command
    command = f"""
classifier prepare-training \\
    --config {config} \\
    --sam2-cfg {sam2_cfg} \\
    --voxel-size {voxel_size} \\
    --tomogram-algorithm {tomogram_algorithm} \\
    --slab-thickness {slab_thickness} \\
    --zarr-path {zarr_path}
    """
    
    if num_slabs > 1:
        command += f" --num-slabs {num_slabs}"

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="prepare-sam2-training",
        output_file="prepare-sam2-training.out",
        shell_name="prepare-sam2-training.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )
