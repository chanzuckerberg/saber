from saber.segmenters.tomo import tomoSegmenter, multiDepthTomoSegmenter
from saber.segmenters.micro import cryoMicroSegmenter
from saber.classifier.models import common
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import SAM2AdapterConfig
import torch


def micrograph_workflow(
    gpu_id:int, cfg:cfgAMG, model_weights:str, model_config:str, target_class:int):
    """Load micrograph segmentation models once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    adapter_cfg = SAM2AdapterConfig(classifier=predictor, amg_cfg=cfg)
    segmenter = cryoMicroSegmenter(cfg=adapter_cfg, deviceID=gpu_id)

    return {
        'segmenter': segmenter,
        'target_class': target_class,
    }

def tomogram_workflow(
    gpu_id:int,
    model_weights:str, model_config:str,
    target_class:int,
    num_slabs:int,
    ):
    """Load tomogram segmentation models once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    cfg_obj = SAM2AdapterConfig(classifier=predictor)
    if num_slabs > 1:
        segmenter = multiDepthTomoSegmenter(cfg=cfg_obj, deviceID=gpu_id, target_class=target_class)
    else:
        segmenter = tomoSegmenter(cfg=cfg_obj, deviceID=gpu_id)

    return {
        'predictor': predictor,
        'segmenter': segmenter,
        'target_class': target_class,
    }

def base_microsegmenter(gpu_id:int, cfg:cfgAMG):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = cryoMicroSegmenter(amg_cfg=cfg, deviceID=gpu_id)
    return {
        'segmenter': segmenter
    }

def base_tomosegmenter(gpu_id:int, cfg:cfgAMG):
    """Load Base SAM2 Model for Preprocessing once per GPU"""

    # Load models
    torch.cuda.set_device(gpu_id)
    segmenter = tomoSegmenter(amg_cfg=cfg, deviceID=gpu_id)
    return {
        'segmenter': segmenter
    }