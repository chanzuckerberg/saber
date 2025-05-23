import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-based classifier for evaluating candidate masks.
    """
    def __init__(self, num_classes, in_channels=2, backbone_type='base', hidden_dim=512):
        super().__init__()
        self.name = self.__class__.__name__
        self.input_mode = 'concatenate'

        # Define a mapping for different ConvNeXt variants
        backbone_map = {
            "tiny": models.convnext_tiny, "small": models.convnext_small,
            "base": models.convnext_base, "large": models.convnext_large
        }

        # Validate backbone selection
        if backbone_type not in backbone_map:
            raise ValueError(f"Invalid backbone type: {backbone_type}. Choose from {list(backbone_map.keys())}.")

        # Load ConvNeXt backbone (without pre-trained weights)
        self.backbone = backbone_map[backbone_type](weights=None).features

        # Modify first convolution layer to accept 2-channel input
        if in_channels != 3:
            out_channels = self.backbone[0][0].out_channels
            self.backbone[0][0] = nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=4, stride=4, 
                                            padding=0, bias=False)

        # Classification head
        in_features = self.backbone[-1][-1].block[-2].out_features
        
        # Classification Head
        self.classifier = nn.Sequential(
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),
            
            # MLP
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),                # Smooth activation function
            nn.Dropout(0.2),          # Dropout for regularization; adjust rate as needed
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for ConvNeXtClassifier.
        """
        x = self.backbone(x)  # Extract high-level features
        x = self.classifier(x)  # Pool and classify
        return x