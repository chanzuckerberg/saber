from typing import List, Dict, Any
import numpy as np

def remove_duplicate_masks(masks: List[Dict[str, Any]], 
                          iou_threshold: float = 0.9,
                          area_threshold: float = 0.9,
                          verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Remove duplicate masks from SAM2 output based on IoU, area, and crop_box similarity.
    
    When duplicates are found, keeps the mask with higher stability_score.
    
    Args:
        masks: List of mask dictionaries from SAM2
        iou_threshold: Minimum IoU to consider masks as duplicates (default 0.99)
        area_threshold: Minimum area similarity ratio to consider as duplicates (default 0.99)
    
    Returns:
        List of unique masks with duplicates removed
    """
    
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def are_masks_duplicate(mask1: Dict[str, Any], mask2: Dict[str, Any]) -> bool:
        """Check if two masks are duplicates based on multiple criteria."""
        
        # Check area similarity
        area1 = mask1['area']
        area2 = mask2['area']
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        iou = calculate_iou(mask1['segmentation'], mask2['segmentation'])

        # Check IoU between actual segmentation masks (most expensive check, do last)
        if area_ratio < area_threshold or iou < iou_threshold:
            return False
    
        return True
    
    # Track which masks to keep
    unique_masks = []
    processed_indices = set()
    
    for i, mask1 in enumerate(masks):
        if i in processed_indices:
            continue
            
        # Find all duplicates of this mask
        duplicate_group = [(i, mask1)]
        
        for j in range(i + 1, len(masks)):
            if j in processed_indices:
                continue
                
            mask2 = masks[j]
            if are_masks_duplicate(mask1, mask2):
                duplicate_group.append((j, mask2))
                processed_indices.add(j)
        
        # From the duplicate group, keep the one with highest stability_score
        if len(duplicate_group) > 1:
            best_mask = max(duplicate_group, 
                          key=lambda x: x[1].get('stability_score', 0))
            unique_masks.append(best_mask[1])
            
            # Optional: print which masks were considered duplicates
            duplicate_indices = [idx for idx, _ in duplicate_group]
            kept_index = best_mask[0]

            if verbose:
                print(f"Found duplicate masks at indices {duplicate_indices}, "
                      f"keeping index {kept_index} with stability_score "
                      f"{best_mask[1].get('stability_score', 0):.4f}")
        else:
            unique_masks.append(mask1)
        
        processed_indices.add(i)
    
    return unique_masks