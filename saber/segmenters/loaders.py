from saber.segmenters.micro import cryoMicroSegmenter
from saber.segmenters.tomo import cryoTomoSegmenter
from saber.classifier.models import common
import torch


def micrograph_workflow(gpu_id:int, model_weights:str, model_config:str, target_class:int, sam2_cfg:str):
    """Load micrograph segmentation models once per GPU"""
    
    torch.cuda.set_device(gpu_id)
    
    # Load models
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    segmenter = cryoMicroSegmenter(
        sam2_cfg=sam2_cfg,
        deviceID=gpu_id,
        classifier=predictor,
        target_class=target_class
    )
    
    return {
        'predictor': predictor,
        'segmenter': segmenter
    }


def tomogram_workflow(gpu_id:int, model_weights:str, model_config:str, target_class:int, sam2_cfg:str):
    """Load tomogram segmentation models once per GPU"""
    
    torch.cuda.set_device(gpu_id)
    
    # Load models
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    segmenter = cryoTomoSegmenter(
        sam2_cfg=sam2_cfg, 
        deviceID=gpu_id,
        classifier=predictor,
        target_class=target_class
    )
    
    return {
        'predictor': predictor,
        'segmenter': segmenter
    }

def base_microsegmenter(gpu_id:int, sam2_cfg:str):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = cryoMicroSegmenter( sam2_cfg=sam2_cfg, deviceID=gpu_id )
    return {
        'segmenter': segmenter
    }

def base_tomosegmenter(gpu_id:int, sam2_cfg:str):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = cryoTomoSegmenter( sam2_cfg=sam2_cfg, deviceID=gpu_id )
    return {
        'segmenter': segmenter
    }