from saber.filters.downsample import FourierRescale2D
from saber.segmenters.base import saber2D
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig, SAM2AdapterConfig
from typing import Any, Optional
import torch

class cryoMicroSegmenter(saber2D):
    def __init__(self,
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 50,
        window_size: int = 256,
        overlap_ratio: float = 0.25,
        cfg: cfgAMG = None,          # kept for backward compatibility
        adapter_cfg: Optional[AdapterConfig] = None,
    ):
        """
        Class for Segmenting Micrographs
        """
        # Convert legacy cfg param to adapter_cfg if needed
        if adapter_cfg is None and cfg is not None:
            adapter_cfg = SAM2AdapterConfig(
                min_mask_area=min_mask_area,
            )
        super().__init__(cfg=adapter_cfg, deviceID=deviceID, classifier=classifier, target_class=target_class, min_mask_area=min_mask_area, window_size=window_size, overlap_ratio=overlap_ratio)

        # Max pixels for single inference
        self.max_pixels = 1280

    @torch.inference_mode()
    def segment(self,
        image0,
        target_class: Optional[int] = None,
        text: Optional[str] = None,
        display: bool = True,
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
            text=text,
            display=display,
            use_sliding_window=use_sliding_window
        )