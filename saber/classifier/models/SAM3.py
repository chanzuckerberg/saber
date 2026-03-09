import saber.classifier.models.common as common
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from saber.utils import io


class SAM3Classifier(nn.Module):
    """
    Mask classifier using the SAM3 image encoder as a frozen backbone.

    Mirrors SAM2Classifier but uses SAM3's vision backbone
    (SAM3VLBackbone.forward_image) for feature extraction.  Features are
    extracted from the sam2_backbone_out compatibility layer, which provides
    the same 256-channel embedding that SAM2's image_embed produces, making
    the projection + classification head directly comparable.

    Feature shape note
    ------------------
    SAM2 produces [B, 256, 64, 64] embeddings (1024px / patch-size 16).
    SAM3 produces [B, 256, 63, 63] embeddings (1008px / stride 16).
    The projection head uses adaptive pooling so it handles both sizes.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dims: int = 256,
        fuse_features: bool = False,
        deviceID: int = 0,
    ):
        super().__init__()
        self.use_fused_features = fuse_features
        self.name = self.__class__.__name__
        self.input_mode = "separate"

        # Device
        if deviceID < 0:
            self.device = torch.device("cpu")
        else:
            self.device = io.get_available_devices(deviceID)

        # Build SAM3 image model (weights downloaded from HuggingFace automatically)
        from sam3.model_builder import build_sam3_image_model

        sam3_model = build_sam3_image_model(
            load_from_HF=True,
            device=str(self.device),
            eval_mode=True,
            enable_segmentation=True,
            # enable_inst_interactivity produces the sam2_backbone_out
            # compatibility layer we need for feature extraction
            enable_inst_interactivity=True,
        )
        self.backbone = sam3_model

        # Freeze the SAM3 backbone weights
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Channel counts mirror SAM2Classifier
        if fuse_features:
            start_channels = 704   # [256 + 32 + 64] * 2 (ROI + RONI, fused with FPN)
        else:
            start_channels = 512   # 256 * 2 (ROI + RONI, image_embed only)

        projection_dims = [hidden_dims, hidden_dims // 2, hidden_dims // 4]

        self.projection = nn.Sequential(
            nn.Conv2d(start_channels, projection_dims[0], kernel_size=1),
            nn.BatchNorm2d(projection_dims[0]),
            nn.PReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(projection_dims[0], projection_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(projection_dims[0]),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(projection_dims[0], projection_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(projection_dims[1]),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_dims[1], 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

        common.initialize_weights(self)

    # ------------------------------------------------------------------
    # PyTorch overrides
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
            self.backbone.to(self.device)
        except StopIteration:
            pass
        return self

    def train(self, mode=True):
        super().train(mode)
        # SAM3 backbone always stays in eval mode
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run SAM3's image backbone on a batch of images and return the
        primary embedding.

        Args:
            x: (B, 3, H, W) float32 tensor in [-1, 1].

        Returns:
            image_embed: (B, 256, ~63, ~63) float32 tensor.
        """
        with torch.no_grad():
            # SAM3 backbone expects (B, 3, H, W) in [-1, 1] — matches our range
            backbone_out = self.backbone.backbone.forward_image(x)

        sam2_out = backbone_out.get("sam2_backbone_out", {})
        # vision_features: the primary embedding ([B, 256, H/stride, W/stride])
        image_embed = sam2_out.get("vision_features")

        if image_embed is None:
            raise RuntimeError(
                "sam2_backbone_out['vision_features'] not found in SAM3 backbone output. "
                "Make sure enable_inst_interactivity=True when building the model."
            )

        return image_embed.to(torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, H, W) — single-channel tomogram slice (float32).
            mask: (B, 1, H_orig, W_orig) — binary mask for ROI/RONI split.

        Returns:
            logits: (B, num_classes).
        """
        # Drop the channel dim: (B, 1, H, W) → (B, H, W)
        x = x[:, 0, ...]

        # Cast to float32 safely (handles bfloat16 autocast)
        autocast_off = (
            torch.autocast(device_type="cuda", enabled=False) if x.is_cuda
            else nullcontext()
        )
        with autocast_off:
            x_np = x.detach().to(torch.float32).cpu().numpy()  # (B, H, W)

        # Grayscale → 3-channel RGB for SAM3's backbone
        # Values already in [-1, 1]; repeat along channel axis
        images_rgb = np.repeat(x_np[:, None, ...], 3, axis=1)  # (B, 3, H, W)
        images_t = torch.as_tensor(images_rgb, dtype=torch.float32, device=self.device)

        # Resize to SAM3's expected resolution (1008×1008)
        images_t = F.interpolate(
            images_t,
            size=(self.backbone.backbone.visual.vit_backbone.img_size,) * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Extract features: (B, 256, ~63, ~63)
        features = self._extract_features(images_t)

        # ROI / RONI split along the channel dimension
        features = self.apply_mask_to_features(features, mask)  # (B, 512, ~63, ~63)

        # Project and classify
        features = self.projection(features)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def apply_mask_to_features(
        self,
        feature_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split feature map into ROI (inside mask) and RONI (outside mask)
        and concatenate along the channel dimension.

        Args:
            feature_map: (B, C, H, W)
            mask: (B, 1, H_orig, W_orig) binary mask.

        Returns:
            (B, 2*C, H, W)
        """
        mask = mask.to(feature_map.device)
        mask_resized = F.interpolate(mask, size=feature_map.shape[2:], mode="nearest")
        inv_mask = 1 - mask_resized
        roi_features = feature_map * mask_resized
        roni_features = feature_map * inv_mask
        return torch.cat([roi_features, roni_features], dim=1)
