from saber.classifier.models import ( SwinTransformer, ConvNeXt, SAM2 )
import torch.nn as nn
import os

# Try to import cryoDinoV2, if not available, pass
try: 
    from saber.classifier.models import ( cryoDinoV2 )
    dinov2_available = True
except:
    dinov2_available = False

def get_classifier_model(backbone, num_classes, model_size, deviceID=0, **kwargs):
    model_map = {
        'ConvNeXt': ConvNeXt.ConvNeXtClassifier,
        'SwinTransformer': SwinTransformer.SwinTransformerClassifier,
        'SAM2': SAM2.SAM2Classifier,
        'cryoDinoV2': cryoDinoV2.DinoV2Classifier if dinov2_available else None
    }
    if backbone not in model_map:
        raise ValueError(f"Unsupported backbone type: {backbone}")
    
    # cryoDinoV2 has a fixed backbone, so omit model_size
    if backbone == 'cryoDinoV2':
        return  model_map[backbone](num_classes=num_classes, **kwargs)
    elif backbone == 'SAM2': # We Need to Pass the Device ID to SAM2 Image Encoder
        kwargs['deviceID'] = deviceID
        return  model_map[backbone](num_classes=num_classes, **kwargs)
    else:
        return  model_map[backbone](num_classes=num_classes, backbone_type=model_size, **kwargs)
    
def get_predictor(model_weights, model_config, deviceID: int = 0):
    """
    Initialize the Predictor with the specified model weights and config.

    Args:
        model_weights (str): Path to the model's weight file (.pth).
        model_config (str): Path to the model's config file (.yaml).
        deviceID (int): Device ID for the predictor.

    Returns:
        Predictor: The initialized predictor.
    """
    from saber.classifier.models.predictor import Predictor

    # If model_weights or model_config is missing, return None
    if model_weights is None or model_config is None:
        predictor = None
    # Check if the model_weights and model_config exist, return the predictor
    else:
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model weights file {model_weights} does not exist.")
        elif not os.path.exists(model_config):
            raise FileNotFoundError(f"Model config file {model_config} does not exist.")
        predictor = Predictor(model_config, model_weights, deviceID = deviceID)

    return predictor

def initialize_weights(model):
    """Initialize the weights of the classification and projection head for training stability."""
    # Initialize classifier layers
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Initialize projection layers (Only for SAM2 and DinoV2)
    if model.name == 'SAM2Classifer' or model.name == 'DinoV2Classifier':
        for m in model.projection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)