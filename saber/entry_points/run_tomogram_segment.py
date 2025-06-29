from saber.segmenters.tomo import cryoTomoSegmenter
import saber.process.slurm_submit as slurm_submit
from saber.classifier.models import common
from saber import io, utilities as utils
from saber.visualization import galleries 
from saber.process import mask_filters
from copick_utils.writers import write
import copick, click, torch, os
import multiprocess as mp
from tqdm import tqdm
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@click.option("--run-id", type=str, required=True, 
              help="Path to Copick Config for Processing Data")            
@slurm_submit.classifier_inputs
@slurm_submit.sam2_inputs
def slab(
    config: str,
    run_id: str, 
    voxel_size: int, 
    tomogram_algorithm: str,
    slab_thickness: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str
    ):
    """
    Segment a single slab of a tomogram.
    """

    # Initialize the Domain Expert Classifier   
    predictor = common.get_predictor(model_weights, model_config)

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)

    # Get Run
    run = root.get_run(run_id)

    # Get Tomogram
    print(f'Getting {tomogram_algorithm} Tomogram with {voxel_size} A voxel size for the associated runID: {run.name}')
    vol = io.get_tomogram(run, voxel_size, tomogram_algorithm)

    # Create an instance of cryoTomoSegmenter
    segmenter = cryoTomoSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,         # if you have a classifier; otherwise, leave as None
        target_class=target_class     # desired target class if using a classifier
    )

    # For 2D segmentation, call segment_image
    masks = segmenter.segment_slab(vol, slab_thickness, display_image=True)

def init_worker():
    """Each process gets assigned a unique GPU based on its position in the pool"""
    
    # Get the worker ID from process identity (1-indexed, so subtract 1)
    worker_id = mp.current_process()._identity[0] - 1
    
    # Store in environment variable (most reliable with spawn)
    os.environ['WORKER_GPU_ID'] = str(worker_id)
    
    # Optional debug print
    print(f"Process {os.getpid()} assigned GPU {worker_id}")

def segment_worker(args):
    """Worker function that uses the process's assigned GPU"""
    root, run_id, voxel_size, tomogram_algorithm, segmentation_name, \
    segmentation_session_id, slab_thickness, display_segmentation, \
    model_weights, model_config, target_class, num_slabs, sam2_cfg = args

    # Get GPU ID from environment variable
    gpu_id = int(os.environ['WORKER_GPU_ID'])
    
    # Optional debug print
    print(f"Process {os.getpid()} processing {run_id} on GPU {gpu_id}")

    return segment_tomogram_separate_process(
        root, run_id, gpu_id, 
        voxel_size, tomogram_algorithm,     
        segmentation_name, segmentation_session_id,
        slab_thickness, display_segmentation,
        model_weights, model_config,
        target_class, num_slabs, sam2_cfg
    )

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.tomogram_segment_commands
@click.option("--run-ids", type=str, required=False, default=None,
              help="Path to Copick Config for Processing Data")
@slurm_submit.classifier_inputs
@click.option("--num-slabs", type=int, required=False, default=1,
              help="Number of Slabs to Segment")
@slurm_submit.sam2_inputs
def tomograms(
    config: str,
    run_ids: str,
    voxel_size: float, 
    tomogram_algorithm: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    model_config: str,
    model_weights: str,
    target_class: int,
    num_slabs: int,
    sam2_cfg: str
    ):
    """
    Generate a 3D Segmentation of a tomogram.
    """

    print(f'\nRunning SAM2 Organelle Segmentations for the Following Tomograms:\n Algorithm: {tomogram_algorithm}, Voxel-Size: {voxel_size} Å')

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)

    # Get RunIDs from Copick Project
    if run_ids is None:
        display_segmentation = False
        run_ids = [run.name for run in root.runs]
    else:
        run_ids = [run_ids]
        display_segmentation = True   
        segment_tomogram_separate_process(
            root,
            run_ids[0],
            0,
            voxel_size,
            tomogram_algorithm,
            segmentation_name,
            segmentation_session_id,
            slab_thickness,
            display_segmentation,
            model_weights,
            model_config,
            target_class,
            num_slabs,
            sam2_cfg
        )
        return

    n_procs = torch.cuda.device_count()
    print(f'Parallelizing the Computation over {n_procs} GPUs\n')    

    # Set the Start Method to Spawn
    mp.set_start_method("spawn")

    # Run Segmentation for Each Tomogram
    with mp.Pool(processes=n_procs, initializer=init_worker) as pool:  # No initargs needed
        with tqdm(total=len(run_ids), desc='Segmenting Tomograms', unit='run') as pbar:
            # Create argument tuples for each task
            tasks = [
                (root, run_id, voxel_size, tomogram_algorithm, segmentation_name,
                segmentation_session_id, slab_thickness, display_segmentation,
                model_weights, model_config, target_class, num_slabs, sam2_cfg)
                for run_id in run_ids
            ]
            
            for _ in pool.imap_unordered(segment_worker, tasks, chunksize=1):
                pbar.update(1)


    # Create a gallery of the tomograms
    galleries.create_png_gallery(
        f'gallery_sessionID_{segmentation_session_id}/frames',
    )

    print('Completed the Orgnalle Segmentations with Cryo-SAM2!')

def segment_tomogram_separate_process(
    root: str,
    runID: str,
    deviceID: int, 
    voxel_size: float, 
    tomogram_algorithm: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    display_segmentation: bool,
    model_weights: str,
    model_config: str,
    target_class: int,
    num_slabs: int,
    sam2_cfg: str
    ):

    # Get Run
    run = root.get_run(runID)

    # Get Tomogram, Return None if No Tomogram is Found
    vol = io.get_tomogram(run, voxel_size, algorithm = tomogram_algorithm)
    if vol is None:
        print(f'No Tomogram Found for {runID}')
        return

    # Initialize the Domain Expert Classifier   
    classifier = common.get_predictor(model_weights, model_config, deviceID)        

    # Initialize the SAM2 Segmenter
    segmenter = cryoTomoSegmenter(
        sam2_cfg = sam2_cfg,
        deviceID = deviceID,
        classifier = classifier,
        target_class = target_class
    )
    
    # Handle multiple slabs
    if num_slabs > 1:

        # Default Showing Segmentation to False
        display_segmentation = False

        # Get the Center Index of the Tomogram
        depth = vol.shape[0]
        center_index = depth // 2
        
        # Initialize combined mask with zeros (using volume shape)
        combined_mask = np.zeros((vol.shape), dtype=np.uint8)

        # Process each slab
        mask_label = 0
        for i in range(num_slabs):
            # Define the center of the slab
            offset = (i - num_slabs // 2) * slab_thickness
            slab_center = center_index + offset
            
            # Segment this slab
            segment_mask = segmenter.segment(
                vol, run, slab_thickness, zSlice=slab_center, 
                save_run=runID + '-' + segmentation_session_id, 
                show_segmentations=display_segmentation)        

            # Process and combine masks immediately if valid
            if segment_mask is not None:
                # Offset non-zero values by the mask label
                mask_copy = segment_mask.copy()
                mask_copy[mask_copy > 0] += mask_label
                combined_mask = np.maximum(combined_mask, mask_copy)
                mask_label += 1

        # Apply Adaptive Gaussian Smoothing to the Segmentation Mask              
        combined_mask = mask_filters.fast_3d_gaussian_smoothing(combined_mask, scale = 0.075, deviceID = deviceID)        

        # Combine masks from all slabs
        segment_mask = mask_filters.merge_segmentation_masks(combined_mask)

    else:
        # Single slab case
        segment_mask = segmenter.segment(
            vol, slab_thickness, 
            save_run=runID + '-' + segmentation_session_id, 
            show_segmentations=display_segmentation)

        # Check if the segment_mask is None
        if segment_mask is None:
            print(f'No Segmentation Found for {runID}')
            return

    # Write Segmentation if We aren't Displaying Results
    if not display_segmentation and segment_mask is not None: 

        # Apply Adaptive Gaussian Smoothing to the Segmentation Mask   
        segment_mask = mask_filters.fast_3d_gaussian_smoothing(segment_mask, scale = 0.05, deviceID = deviceID)
        
        # Convert the Segmentation Mask to a uint8 array
        segment_mask = segment_mask.astype(np.uint8)

        # print('Saving the Segmentation to a MRC File..')
        # import mrcfile
        # mrcfile.write('segment.mrc', segment_mask, overwrite=True) 

        # Write Segmentation to Copick Project
        write.segmentation(
            run, 
            segment_mask,
            'SABER',
            name=segmentation_name,
            session_id=segmentation_session_id,
            voxel_size=float(voxel_size)
        )

    # Clear GPU memory
    del vol
    del segmenter
    del segment_mask
    torch.cuda.empty_cache()
