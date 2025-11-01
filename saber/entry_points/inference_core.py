from saber.segmenters.micro import cryoMicroSegmenter
from saber.filters.downsample import FourierRescale2D
from saber.segmenters.tomo import cryoTomoSegmenter
from saber.filters import masks as mask_filters
from copick_utils.io import writers, readers
from saber.utils import zarr_writer, io
import numpy as np
import torch, os

def segment_tomogram_core(
    run,
    voxel_size: float,
    tomogram_algorithm: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    num_slabs: int, delta_z: int,
    display_segmentation: bool,
    segmenter,  # Pre-loaded or newly created segmenter
    gpu_id: int = 0  # Default GPU ID
    ):
    """
    Core segmentation function that both interactive and parallel versions call.
    
    Args:
        run: Copick run object
        segmenter: Pre-loaded segmenter object
        gpu_id: GPU device ID for processing
        ... (other segmentation parameters)
    
    Returns:
        str: Success message or None if failed
    """
    
    # Get Tomogram, Return None if No Tomogram is Found
    vol = readers.tomogram(run, voxel_size, algorithm=tomogram_algorithm)
    if vol is None:
        print(f'No Tomogram Found for {run.name}')
        return None

    # Ensure we're on the correct GPU
    torch.cuda.set_device(gpu_id)
    
    # Segment the Tomogram
    save_name_name = run.name + '-' + segmentation_session_id
    if num_slabs > 1:
        segment_mask = segmenter.segment(
            vol, slab_thickness, num_slabs, delta_z,
            save_name_name, display_segmentation)
    else:
        segment_mask = segmenter.segment(
            vol, slab_thickness,
            save_run=save_name_name, 
            show_segmentations=display_segmentation)


    # Check if the segment_mask is None
    if segment_mask is None:
        print(f'No Segmentation Found for {run.name}')
        return None

    # Write Segmentation if We aren't Displaying Results
    if not display_segmentation and segment_mask is not None: 
        # Apply Adaptive Gaussian Smoothing to the Segmentation Mask
        segment_mask = mask_filters.fast_3d_gaussian_smoothing(
            segment_mask, scale=0.05, deviceID=gpu_id)
        
        # Convert the Segmentation Mask to a uint8 array
        segment_mask = segment_mask.astype(np.uint8)

        # Write Segmentation to Copick Project
        writers.segmentation(
            run, 
            segment_mask,
            'saber',
            name=segmentation_name,
            session_id=segmentation_session_id,
            voxel_size=float(voxel_size)
        )

        # Print Success Message
        print(f'Saved Segmentation for {run.name} as {segmentation_name}')

    # Clear GPU memory (but keep models if they're pre-loaded)
    del vol
    del segment_mask
    torch.cuda.empty_cache()

    # Reset the Inference State
    segmenter.inference_state = None

    return

def segment_micrograph_core(
    input:str, output: str,
    scale_factor: float, target_resolution: float,
    display_image: bool, use_sliding_window: bool,
    gpu_id, models):

    # Use pre-loaded segmenter
    segmenter = models['segmenter']

    # Get the Global Zarr Writer
    zwriter = zarr_writer.get_zarr_writer(output) 
    zwriter.set_dict_attr('amg', segmenter.cfg)

    # Ensure we're on the correct GPU
    torch.cuda.set_device(gpu_id)
    
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
    segmenter.segment( image, display_image=False, use_sliding_window=use_sliding_window )

    # Convert any numpy array/scalar to Python scalar
    if isinstance(pixel_size, np.ndarray):
        pixel_size = pixel_size.item()    

    # Convert Masks to Numpy Array
    masks = mask_filters.masks_to_array(segmenter.masks)

    # For now let's assume the pixel size is in nanometers
    if pixel_size is not None:
        pixel_size /= 10
    else: 
        pixel_size = 1

    # For now lets assume its always grayscale images
    if image.ndim == 2:
        out_image = segmenter.image[:,:,0]

    # Write Run to Zarr
    input = os.path.splitext(os.path.basename(input))[0]
    zwriter.write(
        run_name=input, image=out_image, 
        masks=masks.astype(np.uint8), pixel_size=pixel_size
    )