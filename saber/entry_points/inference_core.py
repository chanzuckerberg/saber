from saber.segmenters.tomo import cryoTomoSegmenter
from saber.process import mask_filters
from copick_utils.writers import write
from saber import io
import numpy as np
import torch

def segment_tomogram_core(
    run,
    voxel_size: float,
    tomogram_algorithm: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    num_slabs: int,
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
    vol = io.get_tomogram(run, voxel_size, algorithm=tomogram_algorithm)
    if vol is None:
        print(f'No Tomogram Found for {run.name}')
        return None

    # Ensure we're on the correct GPU
    torch.cuda.set_device(gpu_id)
    
    # Handle multiple slabs
    if num_slabs > 1:
        # Default Showing Segmentation to False for multi-slab
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
                vol, slab_thickness, zSlice=slab_center, 
                save_run=run.name + '-' + segmentation_session_id, 
                show_segmentations=display_segmentation)        

            # Process and combine masks immediately if valid
            if segment_mask is not None:
                # Offset non-zero values by the mask label
                mask_copy = segment_mask.copy()
                mask_copy[mask_copy > 0] += mask_label
                combined_mask = np.maximum(combined_mask, mask_copy)
                mask_label += 1

        # Apply Adaptive Gaussian Smoothing to the Segmentation Mask              
        combined_mask = mask_filters.fast_3d_gaussian_smoothing(
            combined_mask, scale=0.075, deviceID=gpu_id)        

        # Combine masks from all slabs
        segment_mask = mask_filters.merge_segmentation_masks(combined_mask)

    else:
        # Single slab case
        segment_mask = segmenter.segment(
            vol, slab_thickness, 
            save_run=run.name + '-' + segmentation_session_id, 
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
        write.segmentation(
            run, 
            segment_mask,
            'SABER',
            name=segmentation_name,
            session_id=segmentation_session_id,
            voxel_size=float(voxel_size)
        )

    # Clear GPU memory (but keep models if they're pre-loaded)
    del vol
    del segment_mask
    torch.cuda.empty_cache()
    
    return f"Successfully processed {run.name}"

