import saber.classifier.models.common as common
import saber.utilities as utils
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, os

# DINOv2 Imports
from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.linear import get_args_parser
from dinov2.utils.config import get_cfg_from_args

class DinoV2Classifier(nn.Module):
    """
    Swin Transformer-based classifier for evaluating candidate masks.
    """
    def __init__(
        self, num_classes, 
        hidden_dims= 256,
        cfg_path='/hpc/projects/group.czii/jonathan.schwartz/lysosomes/cryodino/vitb8_lysosome.yaml', 
        checkpoint_path='/hpc/projects/group.czii/jonathan.schwartz/lysosomes/cryodino/teacher_checkpoint.pth',
        deviceID: int = 0):

        super().__init__()
        self.name = self.__class__.__name__
        self.input_mode = 'separate'

        # Get Device
        self.device = utils.get_available_devices(deviceID)

        # Build DINOv2 model
        args_parser = get_args_parser(description='DINOv2 linear evaluation')
        args = args_parser.parse_args(f'--config-file {cfg_path} --pretrained-weights {checkpoint_path}'.split())
        cfg = get_cfg_from_args(args)
        
        # Todo: Edit this Function so that I can pass in the specified device
        self.backbone = build_model_for_eval(cfg, checkpoint_path)
        
        # Freeze the DinoV2 Weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Project the Features to a lower dimension for the classifier
        projection_dims = [hidden_dims, hidden_dims // 4]
        self.projection = nn.Sequential(
            # First reduce channels
            nn.Conv2d(1536, projection_dims[0], kernel_size=1),
            nn.BatchNorm2d(projection_dims[0]),
            nn.PReLU(),
            nn.Dropout2d(0.05),
            
            # Add spatial reduction with 3x3 conv and max pooling
            nn.Conv2d(projection_dims[0], projection_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(projection_dims[0]),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dims by 2x (32x32)
            nn.Dropout2d(0.1),
            
            # Further reduction
            nn.Conv2d(projection_dims[0], projection_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(projection_dims[1]),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # Reduce to 19 x 19 
            nn.Dropout2d(0.2),
        )

        # Classification head (fully connected layers)
        classifier_dims = [128, 64]
        self.classifier = nn.Sequential(
            nn.Linear(projection_dims[1], classifier_dims[0]),  #  feature dim
            nn.LayerNorm(classifier_dims[0]),
            nn.PReLU(),                           # Smooth activation function
            nn.Dropout(0.1),                     # Dropout for regularization; adjust rate as needed
            
            nn.Linear(classifier_dims[0], classifier_dims[1]),  #  feature dim
            nn.LayerNorm(classifier_dims[1]),
            nn.PReLU(),                           # Smooth activation function
            nn.Dropout(0.1),                     # Dropout for regularization; adjust rate as needed
            
            nn.Linear(classifier_dims[1], num_classes)   # Output classification layer
        )
        
        # Weight initialization for better convergence
        common.initialize_weights(self)

    def train(self, mode=True):
        """
        Override the default train() to ensure the backbone always remains in eval mode.
        """
        super().train(mode)
        # Force the SAM2 backbone into evaluation mode even during training
        self.backbone.eval()
        return self

    def forward(self, x, mask):
        """
        Forward pass for SAM2Classifier.
        Args:
            x: Input tensor of shape [B, 1, H, W]
            mask: Unused in this example or processed later
        """
        
        # Get Feature Tokens  (Stop the Gradient)
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(x, norm=True)[0]
        
        # Permute and Reshape to (B, C, H, W)  - (B, 768, 79, 79)
        features = features.permute(0, 2, 1)
        nPix = int( np.sqrt(features.shape[2]))
        features = features.reshape(x.shape[0], 768, nPix, nPix)
            
        # Apply Mask to Features
        features = self.apply_mask_to_features(features, mask)
        
        # Project features into a lower-dimensional space
        features = self.projection(features)  # now shape: [B, hidden_dims[1] // 4, 19, 19]
        
        # Now pool to create a feature vector.
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        # Multi-Headed Attention Pooling Instead?

        # Classify 
        logits = self.classifier(features)  
        return logits

    def apply_mask_to_features(self, feature_map, mask):
        """
        Applies a binary mask and its inverse to a feature map.
        
        Args:
            feature_map: Tensor of shape (B, C, H, W)
            mask: Binary tensor of shape (B, 1, H_orig, W_orig)
        
        Returns:
            concatenated_features: Tensor of shape (B, 2*C, H, W)
                Where the first C channels correspond to the masked ROI and the next C channels correspond to the background.
        """
        # Resize the mask to the feature map's spatial dimensions
        mask_resized = F.interpolate(mask, size=feature_map.shape[2:], mode='nearest')
        
        # Compute the inverse mask
        inv_mask = 1 - mask_resized
        
        # Apply masks
        roi_features = feature_map * mask_resized
        roni_features = feature_map * inv_mask
        
        # Concatenate along the channel dimension
        concatenated_features = torch.cat([roi_features, roni_features], dim=1)
        return concatenated_features
