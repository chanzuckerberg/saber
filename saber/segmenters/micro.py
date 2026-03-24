from saber.filters.downsample import FourierRescale2D
from saber.segmenters.base import saber2D
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig
from typing import Any, Optional
import torch

class cryoMicroSegmenter(saber2D):
    def __init__(self,
        deviceID: int = 0,
        cfg: Optional[AdapterConfig] = None,
        amg_cfg: Optional[cfgAMG] = None,
        min_mask_area: int = 50,
        window_size: int = 256,
        overlap_ratio: float = 0.25,
    ):
        """
        Class for Segmenting Micrographs
        """
        super().__init__(cfg=cfg, amg_cfg=amg_cfg, deviceID=deviceID,
                         min_mask_area=min_mask_area, window_size=window_size,
                         overlap_ratio=overlap_ratio)

        # Max pixels for single inference
        self.max_pixels = 1280

    @torch.inference_mode()
    def segment(self,
        image0,
        target_class: Optional[int] = None,
        text: Optional[str] = None,
        display: bool = True,
        threshold: Optional[float] = 0.5,
        use_sliding_window: bool = False,
    ):
        """
        Segment image using sliding window approach

        Args:
            image0: Input image
            target_class: Override the classifier target class for this call
            text: Text prompt for SAM3-based segmentation
            display: Whether to display the result
            use_sliding_window: Whether to use sliding window (True) or single inference (False)
        """

        # Store the Original Image
        self.image0 = image0
        (nx, ny) = image0.shape

        # Check to See if We Might Reach Memory Limits
        if (nx > self.max_pixels or ny > self.max_pixels) and not use_sliding_window:
            print(f'Image is Larger than {self.max_pixels} pixels in at least one dimension.\nCurrent Size: ({nx}, {ny})')
            print('Consider Downsampling or Using Sliding Window Inference.')

        return super().segment(
            image0,
            target_class=target_class,
            text=text, threshold=threshold,
            display=display,
            use_sliding_window=use_sliding_window
        )