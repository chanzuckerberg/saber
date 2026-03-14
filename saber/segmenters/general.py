from saber.segmenters.base import saber3D
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig
from typing import Optional
import torch

class volumeSegmenter(saber3D):
    def __init__(self,
        deviceID: int = 0,
        cfg: Optional[AdapterConfig] = None,
        amg_cfg: Optional[cfgAMG] = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025,
    ):
        """
        Initialize the generalSegmenter
        """
        self.target_class = target_class
        self.min_rel_box_size = min_rel_box_size
        super().__init__(
            deviceID=deviceID, cfg=cfg, amg_cfg=amg_cfg,
            min_mask_area=min_mask_area,
        )

    @torch.inference_mode()
    def segment_3d(
        self,
        vol,
        masks,
        ann_frame_idx: int = None,
        show_segmentations: bool = False
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Create Inference State via adapter set_volume()
        if not self._vol_loaded:
            self.video_predictor.set_volume(vol)
            self._vol_loaded = True

        # Set Masks - Right now this is external
        self.masks = masks

        # Get Dimensions
        nx = vol.shape[0]
        ny, nz = self.masks[0].shape[0], self.masks[0].shape[1]

        # Set annotation frame
        self.ann_frame_idx = ann_frame_idx if ann_frame_idx is not None else nx // 2

        # Propagate and filter (adapter handles seeding + hook + scoring internally)
        mask_shape = (nx, ny, nz)
        vol_masks = self.propagate(mask_shape)

        return vol_masks