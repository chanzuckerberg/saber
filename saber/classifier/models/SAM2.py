import saber.classifier.models.common as common
from saber import pretrained_weights
import torch.nn.functional as F
from saber.utils import io
import torch.nn as nn
import numpy as np
import torch

# Suppress Warning for Post Processing from SAM2 - 
# Explained Here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Silence SAM2 loggers
import logging
logging.getLogger("sam2").setLevel(logging.ERROR)  # Only show errors

class SAM2Classifier(nn.Module):
    """
    Mask classifier with the SAM2 Image Embeddings for evaluating candidate masks.
    """
    def __init__(
        self, num_classes, 
        backbone_type="large", 
        hidden_dims=256,
        fuse_features=False,
        deviceID: int = 0):

        super().__init__()
        self.use_fused_features = fuse_features  # Flag to choose feature pipeline
        self.name = self.__class__.__name__
        self.input_mode = 'separate'

        # Get Device
        self.device = io.get_available_devices(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(backbone_type)
        sam2_model = build_sam2(cfg, checkpoint, device=self.device)
        self.backbone = SAM2ImagePredictor(sam2_model)
        
        # Freeze the SAM2 Weights
        self.backbone.model.eval()
        for param in self.backbone.model.parameters():
            param.requires_grad = False
            
        # Determine number of channels after the SAM2 backbone
        if fuse_features:  start_channels = 704     # [256 + 32 + 64] * 2  
        else:              start_channels = 512     # 256 * 2 (ROI + RONI) 
            
        # Project the Features to a lower dimension for the classifier
        projection_dims = [hidden_dims, hidden_dims//2, hidden_dims // 4]        
        self.projection = nn.Sequential(
            # First reduce channels
            nn.Conv2d(start_channels, projection_dims[0], kernel_size=1),
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
            nn.MaxPool2d(2, 2),  # Reduce to 16x16
            nn.Dropout2d(0.2),            
        )

        # Current issue, we're overfitting since we still have a 64x64 feature map.
        # We should try to reduce the spatial dimensions of the feature map.
        # We can do this by adding a MaxPool2d layer after the projection.
        # Lets reduce the spatial dimensions by 4x. to 8x8 

        # Classification head (fully connected layers)
        # classifier_dims = [256, 128]
        self.classifier = nn.Sequential(
            nn.Linear(projection_dims[1], 64),  #  feature dim
            nn.LayerNorm(64),
            nn.PReLU(),                           # Smooth activation function
            nn.Dropout(0.1),                     # Dropout for regularization; adjust rate as needed
            nn.Linear(64, num_classes)   # Output classification layer
        )
        
        # Weight initialization for better convergence
        common.initialize_weights(self)

    def train(self, mode=True):
        """
        Override the default train() to ensure the backbone always remains in eval mode.
        """
        super().train(mode)
        # Force the SAM2 backbone into evaluation mode even during training
        self.backbone.model.eval()
        return self

    def forward(self, x, mask):
        """
        Forward pass for SAM2Classifier.
        Args:
            x: Input tensor of shape [B, 1, H, W]
            mask: Unused in this example or processed later
        """
        # Remove the channel dimension:
        x = x[:, 0, ...]  # Now x has shape [B, H, W]

        # Move to CPU and convert to numpy
        x_np = x.cpu().numpy()  # shape: [B, H, W]

        # Convert each grayscale image to a 3-channel image (H, W, 3)
        images_list = [np.repeat(img[..., None], 3, axis=2) for img in x_np]
        
        # Process the batch
        with torch.no_grad():
            self.backbone.set_image_batch(images_list)  # this calls reset_predictor() internally
        features = self.backbone._features["image_embed"]  # [B, 256, 64, 64]
        
        # # Fuse Global and/or High-Resolution Features
        # if self.use_fused_features:
        #     high_res_feats = self.backbone._features["high_res_feats"]  # [[B, 32, 256, 256], [B, 64, 128, 128]]
        #     fused = self.fuse_features(embed, high_res_feats)    
        # else:
        #     fused = embed
            
        # Apply Mask to Features
        features = self.apply_mask_to_features(features, mask)
        
        # Project features into a lower-dimensional space
        features = self.projection(features)  # now shape: [B, hidden_dims[1] // 4, 16, 16]
        
        # Now pool to create a feature vector.
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1) # shape: [B, hidden_dims[1] // 4]

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
    
    
    def fuse_features(self,image_embed, high_res_feats):
        """
        Given:
        image_embed: a tensor of shape (B, C, H, W) from the SAM2 backbone (global features)
        high_res_feats: a list of tensors of shape (B, C_i, H_i, W_i) from intermediate layers (finer features)
        
        Returns:
        fused_features: a tensor of shape (B, fused_dim)
        """
        # Global feature vector via adaptive average pooling on image_embed
        global_feat = F.adaptive_avg_pool2d(image_embed, (1, 1)).view(image_embed.size(0), -1)
        
        # Process each high-res feature: pool to (1,1) and flatten
        pooled_high_res = [
            F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
            for feat in high_res_feats
        ]
        
        # Concatenate all features into a single vector
        fused_features = torch.cat([global_feat] + pooled_high_res, dim=1)
        return fused_features