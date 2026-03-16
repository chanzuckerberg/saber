from saber.visualization import classifier as viz
import saber.filters.masks as filters
from saber.adapters.base import AdapterConfig, SAM2AdapterConfig, get_adapter
from saber.utils import preprocessing as prep
from saber.adapters.sam2.amg import cfgAMG
from typing import List, Tuple, Optional
from saber.segmenters import utils
from saber.utils import io
from tqdm import tqdm
import numpy as np
import torch

# Suppress SAM2 Logger 
import logging
logger = logging.getLogger()
logger.disabled = True

class saber2D:
    def __init__(self,
        deviceID: int = 0,
        cfg: Optional[AdapterConfig] = None,
        amg_cfg: Optional[cfgAMG] = None,
        min_mask_area: int = 50,
        window_size: int = 256,
        overlap_ratio: float = 0.25,
    ):
        """
        Class for Segmenting Micrographs or Images
        """
        if cfg is None and amg_cfg is None:
            raise "Either Provide an AdapterConfig or AMG Config!"

        # Wrap bare amg_cfg into a full config
        if cfg is None:
            cfg = SAM2AdapterConfig(amg_cfg=amg_cfg, min_mask_area=min_mask_area)

        # Minimum Mask Area to Ignore
        self.min_mask_area = min_mask_area

        # Sliding window parameters
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        # Determine device
        self.device = io.get_available_devices(deviceID)
        self.deviceID = deviceID

        # Initialize Domain Expert Classifier for Filtering False Positives
        _classifier = getattr(cfg, 'classifier', None)
        if _classifier is None:
            self.classifier = None
            self.batchsize = None     
        else:
            self.classifier = _classifier
            self.batchsize = 32 

        self.adapter_cfg = cfg
        self.adapter = get_adapter(cfg, self.device)

        # Initialize Image and Masks
        self.image = None

        # Internal Variable to Let Users Save Segmentations
        self.save_button = False
        self.remove_repeating_masks = True

    def segment(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        text: Optional[str] = None,
        threshold: Optional[float] = 0.5,
        display: bool = False,
        use_sliding_window: bool = False,
    ) -> list:
        return self.segment_image(
            image, display=display,
            use_sliding_window=use_sliding_window,
            text_prompt=text, threshold=threshold,
            target_class=target_class,
        )

    @torch.inference_mode()
    def segment_image(self,
        image: np.ndarray,
        display: bool = True,
        use_sliding_window: bool = False,
        text_prompt: Optional[str] = None,
        threshold: Optional[float] = 0.5,
        target_class: Optional[int] = 1,
    ):
        """
        Segment image using sliding window approach
        
        Args:
            image: Input image
            display: Whether to display the result
            use_sliding_window: Whether to use sliding window (True) or single inference (False)
        """
        
        # Run Segmentation
        self.target_class = target_class
        if use_sliding_window:

            # Create Full Mask
            full_mask = np.zeros(image.shape[:2], dtype=np.uint16)

            # Get sliding windows
            windows = self.get_sliding_windows(image.shape)
            
            # Process each window
            all_masks = []
            for i, (y1, x1, y2, x2) in tqdm(enumerate(windows), total=len(windows)):
                # Extract window
                window_image = image[y1:y2, x1:x2]
                
                # Run inference on window
                window_masks = self.adapter.segment_image_2d(
                    window_image, text_prompt=text_prompt, 
                    threshold=threshold 
                )
                
                # Transform masks back to full image coordinates
                curr_masks = []
                for mask in window_masks:

                    # Filter Out Small Masks
                    if mask['area'] < self.min_mask_area:
                        continue
                    
                    # IMPORTANT: leave mask['segmentation'] as the SMALL local bool array
                    mask['offset'] = (y1, x1)
                    mask['bbox'] = self._to_global_bbox(mask['bbox'], y1, x1)

                    curr_masks.append(mask)

                # Apply Classifier to Filter False Positives
                all_masks.extend( self._apply_classifier(window_image, curr_masks) )

            # Store the Masks
            self.masks = self.rasterize_masks(image, all_masks)
            
        else:
            # Original single inference
            self.masks = self.adapter.segment_image_2d(
                image, text_prompt=text_prompt, threshold=threshold)

            # Apply Classifier to Filter False Positives
            self.masks = self._apply_classifier(image, self.masks)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display:
            viz.display_mask_list(image, self.masks, self.save_button)

        # Return the Masks
        self.image = image
        return self.masks  

    def _apply_classifier(self, image, masks):

        # Filter out small masks + Remove Repeating Masks if Desired
        masks = [mask for mask in masks if mask['area'] >= self.min_mask_area]
        if self.remove_repeating_masks:
            masks = utils.remove_duplicate_masks(masks)

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is None:
            # Since Order Doesn't Matter, Sort by Area for Saber GUI. 
            masks = sorted(masks, key=lambda mask: mask['area'], reverse=False)
        else: 
            gray = image[:, :, 0] if image.ndim == 3 else image
            masks = filters.apply_classifier(
                gray, masks, self.classifier,
                self.target_class, self.batchsize)

        return masks
        
    def get_sliding_windows(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding window coordinates
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            List of (y1, x1, y2, x2) coordinates for each window
        """
        h, w = image_shape[:2]
        stride = int(self.window_size * (1 - self.overlap_ratio))
        
        windows = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = y
                x1 = x
                y2 = min(y + self.window_size, h)
                x2 = min(x + self.window_size, w)
                
                # Skip windows that are too small
                if (y2 - y1) < self.window_size // 2 or (x2 - x1) < self.window_size // 2:
                    continue
                    
                windows.append((y1, x1, y2, x2))
                
        return windows

    def _to_global_bbox(self, local_bbox, y0, x0):
        # SAM-style bbox = [x, y, w, h]
        x, y, w, h = local_bbox
        return [x + x0, y + y0, w, h]

    def rasterize_masks(self, image, masks):
        """
        Convert local masks to full-res binary overlays (only when needed).
        Returns a shallow-copied list with 'segmentation' replaced by full-sized arrays.
        """
        H, W = image.shape[:2]
        disp = []
        for m in masks:
            y0, x0 = m['offset']
            seg = m['segmentation']
            h, w = seg.shape
            full = np.zeros((H, W), dtype=bool)
            y1, x1 = max(0, y0), max(0, x0)
            y2, x2 = min(H, y0 + h), min(W, x0 + w)
            sy1, sx1 = y1 - y0, x1 - x0
            sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
            full[y1:y2, x1:x2] = seg[sy1:sy2, sx1:sx2]
            m2 = dict(m)
            m2['segmentation'] = full
            disp.append(m2)
        return disp
    
class saber3D(saber2D):
    def __init__(self,
        deviceID: int = 0,
        cfg: AdapterConfig = None,
        amg_cfg: cfgAMG = None,
        min_mask_area: int = 50,
    ):
        super().__init__(
            deviceID=deviceID,
            cfg=cfg, amg_cfg=amg_cfg,
            min_mask_area=min_mask_area,
        )

        # Alias for downstream 3D segmentation code
        self.video_predictor = self.adapter

        # Track whether volume has been loaded into the adapter
        self._vol_loaded = False

        # Minimum Logits Threshold for Confidence
        self.min_logits = 0.5

        # Flag to Plot the Z-Slice Confidence Estimations
        self.confidence_debug = False

        # Default to full volume propagation
        self.nframes = None

        # Filter Threshold for Confidence
        self.filter_threshold = 0.5
        
    def propagate(self, mask_shape, target_class: Optional[int] = 1):
        """Seed masks into the adapter and propagate bidirectionally."""
        if isinstance(self.masks[0], dict):
            mask_arrays = [m['segmentation'] for m in self.masks]
        else:
            mask_arrays = self.masks

        vol_masks = self.video_predictor.segment_volume(
            start_frame_idx=self.ann_frame_idx,
            masks=mask_arrays,
            vol_shape=mask_shape,
            max_frame_num_to_track=self.nframes,
            min_presence_score=self.filter_threshold,
        )
        self.video_predictor.reset_state()
        return vol_masks
