import torch
import torch.nn as nn
from torchvision import models 
from saber.classifier.models import common

class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer-based classifier for evaluating candidate masks.
    """
    def __init__(
        self, num_classes, 
        in_channels=2, 
        backbone_type="small", 
        hidden_dim=256,
        deviceID: int = 0):

        super().__init__()
        self.name = self.__class__.__name__

        self.input_mode = 'concatenate'
        if backbone_type == 'large':
            ValueError("Large model is not supported for classifier training.")

        # Define a mapping for different Swin Transformer variants
        backbone_map = {
            "tiny": models.swin_v2_t, 
            "small": models.swin_v2_s,
            "base": models.swin_v2_b
        }

        # Validate backbone selection
        if backbone_type not in backbone_map:
            raise ValueError(f"Invalid backbone type: {backbone_type}. Choose from {list(backbone_map.keys())}.")

        # Load Swin Transformer backbone
        self.backbone = backbone_map[backbone_type](weights=None)

        # Modify input embedding layer (Patch Embedding) to accept in_channels
        if in_channels != 3:
            out_channels = self.backbone.features[0][0].out_channels
            self.backbone.features[0][0] = nn.Conv2d(in_channels, out_channels, 
                                                      kernel_size=4, stride=4, 
                                                      padding=0, bias=False)

        # Swin Transformer outputs `(B, N, D)`, where D is feature dim
        in_features = self.backbone.head.out_features 

        # Classification head (fully connected layers)
        self.classifier = nn.Sequential(

            nn.Linear(in_features, hidden_dim),  # Swin Transformer feature dim
            nn.LayerNorm(hidden_dim),
            nn.GELU(),                           # Smooth activation function
            nn.Dropout(0.2),                     # Dropout for regularization; adjust rate as needed
            nn.Linear(hidden_dim, num_classes)   # Output classification layer
        )
        
        # Weight initialization for better convergence
        common.initialize_weights(self)

    def forward(self, x):
        """
        Forward pass for SwinTransformerClassifier.
        """
        x = self.backbone(x)  # Extract high-level features (B, N, D)
        x = self.classifier(x)  # Classify
        return x