from typing import Dict, Any, Optional

# -----------------------------
# Default parameters for SAM2 AMG
# -----------------------------
def get_default() -> Dict[str, Any]:
    """
    Get Default Automatic Mask Generator Parameters for SAM2.
    
    Returns:
        dict: Default parameters for SAM2 Automatic Mask Generator
    """
    return {
        'npoints': 32,
        'points_per_batch': 64,
        'pred_iou_thresh': 0.7,
        'stability_score_thresh': 0.92,
        'stability_score_offset': 0.7,
        'crop_n_layers': 2,
        'box_nms_thresh': 0.7,
        'crop_n_points_downscale_factor': 2,
        'use_m2m': True,
        'multimask_output': True
    }

