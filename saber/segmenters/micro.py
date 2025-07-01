from saber.segmenters.base import saber2Dsegmenter
from saber.process.downsample import FourierRescale2D
from saber.visualization import classifier as viz
import saber.process.mask_filters as filters
from saber import pretrained_weights
import saber.utilities as utils
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch


class cryoMicroSegmenter(saber2Dsegmenter):
    def __init__(self,
        sam2_cfg: str, 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 50,
        window_size: int = 256,
        overlap_ratio: float = 0.25,
    ):
        """
        Class for Segmenting Micrographs
        """

        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, window_size, overlap_ratio)

    @torch.inference_mode()
    def segment(self,
        image0,
        display_image: bool = True,
        use_sliding_window: bool = False
    ):
        """
        Segment image using sliding window approach
        
        Args:
            image0: Input image
            display_image: Whether to display the result
            use_sliding_window: Whether to use sliding window (True) or single inference (False)
        """

        # Store the Original Image
        self.image0 = image0
        (nx, ny) = image0.shape

        # (Optional)Fourier Crop the Image to the Desired Resolution
        if not use_sliding_window and (nx > 1536 or ny > 1536):
            scale_factor =  max(nx, ny) / 1024 
            self.image0 = FourierRescale2D.run(self.image0, scale_factor)
            (nx, ny) = self.image0.shape

        # Increase Contrast of Image and Normalize the Image to [0,1]        
        self.image0 = utils.contrast(self.image0, std_cutoff=2)
        self.image0 = utils.normalize(self.image0, rgb = False)

        # Extend From Grayscale to RGB 
        self.image = np.repeat(self.image0[..., None], 3, axis=2)   

        # Segment Image
        self.segment_image(
            display_image = display_image, 
            use_sliding_window = use_sliding_window)

        return self.masks