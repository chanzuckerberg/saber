from saber.segmenters.tomo import cryoTomoSegmenter, multiDepthTomoSegmenter
from saber.segmenters.micro import cryoMicroSegmenter
from saber.classifier.models import common
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig
from typing import Any, Optional
import torch


def micrograph_workflow(
    gpu_id:int, cfg:cfgAMG, model_weights:str, model_config:str, target_class:int):
    """Load micrograph segmentation models once per GPU"""
    
    # Load models
    torch.cuda.set_device(gpu_id)
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    segmenter = cryoMicroSegmenter(
        cfg=cfg,
        deviceID=gpu_id,
        classifier=predictor,
        target_class=target_class
    )
    
    return {
        'segmenter': segmenter
    }

def tomogram_workflow(
    gpu_id:int,
    model_weights:str, model_config:str,
    target_class:int,
    num_slabs:int,
    adapter_cfg: Optional[AdapterConfig] = None,
    ):
    """Load tomogram segmentation models once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    if num_slabs > 1:
        segmenter = multiDepthTomoSegmenter(
            deviceID=gpu_id,
            classifier=predictor,
            target_class=target_class,
            adapter_cfg=adapter_cfg,
        )
    else:
        segmenter = cryoTomoSegmenter(
            deviceID=gpu_id,
            classifier=predictor,
            target_class=target_class,
            adapter_cfg=adapter_cfg,
        )
    
    return {
        'predictor': predictor,
        'segmenter': segmenter
    }

def base_microsegmenter(gpu_id:int, cfg:cfgAMG):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = cryoMicroSegmenter( cfg=cfg, deviceID=gpu_id )
    return {
        'segmenter': segmenter
    }

def base_tomosegmenter(gpu_id:int, cfg:cfgAMG):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = cryoTomoSegmenter( cfg=cfg, deviceID=gpu_id )
    return {
        'segmenter': segmenter
    }