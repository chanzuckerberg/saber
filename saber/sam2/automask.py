from typing import Dict, Any, Optional
import rich_click as click

# Silence SAM2 loggers
import logging
logging.getLogger("sam2").setLevel(logging.ERROR)  # Only show errors

# Decorator to add common options to a Click command.
def amg_cli():
    
    def decorator(f):
        f = click.option('-cfg', '--sam2-cfg', required=False, default='small', help="SAM2 Model Config",
                         type=click.Choice(['large', 'base', 'small', 'tiny'], case_sensitive=False))(f)
        f = click.option('-npts', '--npoints', type=int, default=32, help='Number of points per side')(f)
        f = click.option('-nbatch', '--points-per-batch', type=int, default=64, help='Number of points per batch')(f)
        f = click.option('-iou', '--pred-iou-thresh', type=float, default=0.7, help='Prediction IOU threshold')(f)
        f = click.option('-nlayers', '--crop-n-layers', type=int, default=2, help='Number of crop layers')(f)
        f = click.option('-box', '--box-nms-thresh', type=float, default=0.7, help='Box NMS threshold')(f)
        f = click.option('-crop', '--crop-n-points', type=int, default=2, help='Crop N Points Downscale Factor')(f)
        f = click.option('-m2m', '--use-m2m', type=bool, default=True, help='Use M2M')(f)
        f = click.option('-multi', '--multimask', type=bool, default=True, help='Multimask Output')(f)
        return f
    return decorator

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


def build_amg(amg_params: str, min_mask_area: int, device: Optional[str] = 'cpu'):

    # Suppress Warning for Post Processing from SAM2 - 
    # Explained Here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from saber import pretrained_weights
    from saber.sam2 import amg as fmask

    # Build SAM2 model
    (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(amg_params['sam2_cfg'])
    sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing = True)
    sam2.eval()

    # Build Mask Generator      
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=amg_params['npoints'],        
        points_per_batch=amg_params['points_per_batch'],            
        pred_iou_thresh=amg_params['pred_iou_thresh'],
        stability_score_thresh=amg_params['stability_score_thresh'],
        stability_score_offset=amg_params['stability_score_offset'],
        crop_n_layers=amg_params['crop_n_layers'],                # 1
        box_nms_thresh=amg_params['box_nms_thresh'],
        crop_n_points_downscale_factor=amg_params['crop_n_points_downscale_factor'],
        use_m2m=amg_params['use_m2m'],
        multimask_output=amg_params['multimask_output'],
    )  

    # Add Mask Filtering to Generator
    mask_generator = fmask.FilteredSAM2MaskGenerator(
        base_generator=mask_generator,
        min_area_filter=min_mask_area,
    )

    return mask_generator