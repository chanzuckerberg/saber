from saber.filters.gaussian import gaussian_smoothing_3d
from saber.utils import io
from scipy import ndimage
import torch, gc, skimage
import numpy as np


def apply_classifier(image, masks, classifier, desired_class: int = None, min_mask_area: int = 100):
    """
    Apply a domain expert classifier to a segmentation mask and return the masks that match the desired class.
    """

    # Extract the 'segmentation' array from each SAM2 mask and convert booleans to uint8 (0, 1)
    sam2_masks = [mask_info['segmentation'].astype(np.uint8) for mask_info in masks]
    sam2_masks = np.array(sam2_masks)

    # Run predictions using your classifier and Determine predicted class for each mask
    with torch.no_grad():
        predictions = classifier.predict(image[:,:,0], sam2_masks)

    return convert_predictions_to_masks(predictions, masks, desired_class, min_mask_area)   

def convert_predictions_to_masks(predictions, masks, desired_class: int = None, min_mask_area: int = 100):

    if isinstance(masks, np.ndarray):
        masks = masks_to_list(masks)

     # Get predicted class for each mask
    predicted_classes = np.argmax(predictions, axis=1)  

    # Instance Segmentation 
    # If a desired class is specified, filter the masks that match the desired class
    if desired_class > 0 and desired_class is not None:

        # Get confidence for the desired class
        confidence_scores = predictions[:,desired_class]

        # Filter masks that match the desired class
        target_indices = [i for i, pred in enumerate(predicted_classes) if pred == desired_class]
        masks = [masks[i] for i in target_indices]
        confidence_scores = confidence_scores[target_indices]

        # Apply Consensus-Based Resolution to the Target Masks
        if len(masks) > 0:
            # Filter Out Small Masks and Apply Consensus-Based Resolution
            masks = _consensus_based_resolution(masks[0]['segmentation'].shape, masks, confidence_scores)
            
            # Sort Masks by Area
            masks = [mask for mask in masks if mask['area'] >= min_mask_area] 
            masks = sorted(masks, key=lambda x: x['area'], reverse=False)
            
    # Semantic Segmentation
    else:
        # Get the shape from the first mask
        if len(masks) > 0:
            nx, ny = masks[0]['segmentation'].shape
        else:
            return np.array([])  # Return empty array if no masks

        masks = _semantic_segmentation(masks, predictions)

    return masks

def _consensus_based_resolution(image_shape, masks, confidences):
    """
    Implements consensus-based approach by combining overlapping masks.
    """
    h, w = image_shape
    
    # Create confidence-weighted mask map
    confidence_map = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int32)
    
    # Accumulate all masks
    for mask_dict, conf in zip(masks, confidences):
        seg = mask_dict['segmentation']
        confidence_map += seg * conf
        overlap_count += seg
    
    # Create consensus regions (where any mask exists)
    consensus_mask = overlap_count > 0
    
    # Normalize confidence by overlap count
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_confidence = np.divide(confidence_map, overlap_count)
        avg_confidence = np.nan_to_num(avg_confidence)
    
    # Label connected components in the consensus mask
    labeled_mask, num_components = ndimage.label(consensus_mask)
    
    # Create new mask dictionaries for each consensus region
    consensus_masks = []
    for label in range(1, num_components + 1):
        component_mask = labeled_mask == label
        
        # # Skip very small components
        # if np.sum(component_mask) < 100:  # Minimum area threshold
        #     continue
        
        # Calculate average confidence in this component
        component_confidence = np.mean(avg_confidence[component_mask])
        
        # Find bounding box
        coords = np.where(component_mask)
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Create new mask dictionary
        new_mask = {
            'segmentation': component_mask,
            'area': int(np.sum(component_mask)),
            'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
            'predicted_iou': float(component_confidence),  # Use confidence as IoU proxy
            'point_coords': [[int((x_min + x_max) / 2), int((y_min + y_max) / 2)]],
            'stability_score': float(component_confidence),  # Use confidence as stability score
            'crop_box': [int(x_min), int(y_min), int(x_max), int(y_max)]
        }
        
        consensus_masks.append(new_mask)
    
    return consensus_masks    

def _semantic_segmentation(masks, predictions):
    """
    Get array that returns the masks as semantic segmentation.
    Merges all masks belonging to each predicted class.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    max_class = np.max(predicted_classes)
    
    # Initialize empty masks for each class (excluding class 0)
    output_masks = []
    for ii in range(1, max_class + 1):  # Start from 1 to skip background class 0
        output_masks.append({
            'segmentation': np.zeros(masks[0]['segmentation'].shape, dtype=np.uint8),
            'area': 0,
        })

    # Merge masks for each class
    for ii in range(len(masks)):
        predicted_class = predicted_classes[ii]
        if predicted_class > 0:  # Skip background class 0
            class_idx = predicted_class - 1  # Adjust index since we start from class 1
            
            # Merge segmentation masks using logical OR
            output_masks[class_idx]['segmentation'] = np.logical_or(
                output_masks[class_idx]['segmentation'], 
                masks[ii]['segmentation']
            ).astype(np.uint8)
            
            # Accumulate area
            output_masks[class_idx]['area'] += masks[ii]['area']
    
    return output_masks

def masks_to_array(mask_list):
    """
    Convert list of masks to single label matrix
    """
    if not isinstance(mask_list, list):
        print('Returning None')
        return None

    # Convert Masks to Numpy Array 
    (nx, ny) = mask_list[0]['segmentation'].shape
    masks = np.zeros([len(mask_list), nx, ny], dtype=np.uint8)

    # Populate the numpy array
    for j, mask in enumerate(mask_list):
        masks[j] = mask['segmentation'].astype(np.uint8) * (j + 1)
    
    return masks

def masks_to_list(masks):
    """
    Convert a 3D mask array to a list of masks.
    """

    # If already a list, return original masks
    if isinstance(masks, list):
        return masks

    # Convert masks to list of dictionaries
    masks_list = []
    vals = np.unique(masks)
    for val in vals:
        mask = masks == val
        masks_list.append({
            'segmentation': mask,
            'area': np.sum(mask > 0)})
    
    return masks_list

def segments_to_mask(video_segments, masks, mask_shape, nMasks):

    for frame_idx in list(video_segments):
        for jj in range(nMasks):
            resized_mask = skimage.transform.resize(
                video_segments[frame_idx][jj+1][0,], 
                (mask_shape[1], mask_shape[2]), anti_aliasing=False
            )
            mask_update = resized_mask > 0
            masks[frame_idx,:,:][mask_update] = jj + 1    

    return masks

def merge_segmentation_masks(segmentation, min_volume_threshold=100):
    """
    Process a 3D segmentation array where different segments have unique labels.
    Merge overlapping regions using OR operation, filter small objects, and relabel consecutively.
    
    Args:
        segmentation: 3D numpy array where different segments have unique label values
        min_volume_threshold: Minimum volume (in voxels) for an object to be kept
        
    Returns:
        Filtered segmentation mask with consecutive labels, where overlapping regions are merged
    """
    if segmentation is None:
        return None
    
    # Get unique labels, excluding background (0)
    unique_labels = np.unique(segmentation)
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]
    
    # Create a binary mask for all objects
    combined_binary = np.zeros_like(segmentation, dtype=bool)
    for label in unique_labels:
        combined_binary = np.logical_or(combined_binary, segmentation == label)
    
    # Label the connected components in the combined binary mask
    labeled_mask, num_features = ndimage.label(combined_binary)
    
    # Filter objects by volume and keep track of properties
    region_properties = []
    for label in range(1, num_features + 1):
        # Create a binary mask for this label
        mask = (labeled_mask == label)
        
        # Calculate volume (number of voxels)
        volume = np.sum(mask)
        
        # Skip objects smaller than minimum volume threshold
        if volume < min_volume_threshold:
            continue
        
        # Store properties
        region_properties.append({
            'label': label,
            'volume': volume
        })
    
    # Sort regions by volume (largest first)
    region_properties.sort(key=lambda x: x['volume'], reverse=True)
    
    # Create new segmentation with consecutively labeled objects
    new_segmentation = np.zeros_like(segmentation)
    
    # Assign new consecutive labels
    for new_label, prop in enumerate(region_properties, 1):
        old_label = prop['label']
        new_segmentation[labeled_mask == old_label] = new_label
    
    return new_segmentation

def fast_3d_gaussian_smoothing(volume, scale=0.075, deviceID = None):
    """
    Apply fast 3D Gaussian smoothing to a volume.
    
    This function takes either a binary volume or a labeled segmentation and applies
    efficient Gaussian smoothing with automatic sigma estimation based on volume.
    
    Args:
        volume: 3D numpy array (binary mask or segmentation with integer labels)
        sigma: Fixed sigma value to use (overrides automatic estimation if provided)
        scale: Scale factor for automatic sigma estimation (ignored if sigma is provided)
        
    Returns:
        Smoothed 3D volume with the same shape as input
    """
    
    # Check if CUDA is available
    device = io.get_available_devices(deviceID)
    
    # Check if input is 3D
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D input, got {volume.ndim}D")
    
    # Check if input is binary or multi-label
    unique_values = np.unique(volume)
    
    # For segmentation with multiple labels
    result = np.zeros_like(volume, dtype=np.uint8)
    
    # Get unique labels (excluding 0 which is typically background)
    labels = unique_values
    if 0 in labels:
        labels = labels[labels != 0]
    
    # Apply Gaussian Smoothing to Each Label
    for label in labels:
        # Create binary mask for this label
        binary_mask = (volume == label)
        
        # Estimate appropriate sigma for this specific label
        label_sigma = _estimate_feature_size_3d(binary_mask, scale)
        
        # Apply 3D Gaussian filter
        smoothed = gaussian_smoothing_3d(binary_mask, label_sigma, device)
        
        # Threshold and add to result
        smoothed = (smoothed > 0.5)
        result[smoothed] = label

        # Explicitly delete intermediate results
        del binary_mask, smoothed
        # Force garbage collection
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return result

def _estimate_feature_size_3d(binary_volume, scale=0.075):
    """
    Estimate feature size for a 3D binary volume based on its volume.
    
    Args:
        binary_volume: 3D binary numpy array
        scale: Scale factor for sigma calculation
        
    Returns:
        Estimated sigma value
    """
    
    # Count total volume (number of True voxels)
    volume = np.sum(binary_volume)
    
    # For 3D: approximate as sphere
    # Volume of sphere = (4/3) * pi * r^3, so diameter = 2 * r = 2 * ((3V)/(4π))^(1/3)
    approx_diameter = 2 * ((3 * volume) / (4 * np.pi))**(1/3)
    
    # Scale to get sigma
    sigma = scale * approx_diameter
    return sigma

# def _semantic_segmentation(masks, predictions):
#     """
#     Get array that returns the masks as semantic segmentation.
#     """

#     # Get predicted class and confidence for each mask
#     predicted_classes = np.argmax(predictions, axis=1)
#     confidence_scores = np.max(predictions, axis=1)

#     # Create semantic segmentation
#     output_masks = np.zeros(masks[0]['segmentation'].shape, dtype=np.uint8)
#     confidence_map = np.zeros(masks[0]['segmentation'].shape, dtype=np.float32)

#     for ii in range(len(masks)):
#         if predicted_classes[ii] > 0:
#             mask = masks[ii]['segmentation']
#             confidence = confidence_scores[ii]
            
#             # Only update pixels where this mask has higher confidence
#             update_pixels = mask & (confidence > confidence_map)
            
#             output_masks[update_pixels] = predicted_classes[ii]
#             confidence_map[update_pixels] = confidence

#     return output_masks


# def overlap(img1, img2):
#     ints = np.logical_and(img1, img2).sum()
#     return ints / min(img1.sum(), img2.sum())

# def calculate_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return intersection / union if union != 0 else 0    

# def convert_mask_array_to_list(mask_array):
#     """
#     Convert a 3D mask array to a list of masks.
#     """
#     masks = []
#     nMasks = mask_array.shape[0]
#     for iMask in range(nMasks):
#         mask = {
#             'segmentation': mask_array[iMask],
#             'area': np.sum(mask_array[iMask]),
#         }
#         masks.append(mask)
#     return masks
