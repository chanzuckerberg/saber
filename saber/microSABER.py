from saber.process.downsample import FourierRescale2D
from saber.visualization import classifier as viz
import saber.process.mask_filters as filters
from saber import pretrained_weights
import saber.utilities as utils
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch


# Suppress Warning for Post Processing from SAM2 - 
# Explained Here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from saber.sam2 import filtered_automatic_mask_generator as fmask
from sam2.build_sam import build_sam2

# Silence SAM2 loggers
import logging
logging.getLogger("sam2").setLevel(logging.ERROR)  # Only show errors

class cryoMicroSegmenter:
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
        Class for Segmenting Micrographs or Images using SAM2
        """

        # Minimum Mask Area to Ignore 
        self.min_mask_area = min_mask_area

        # Sliding window parameters
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = 0.5        

        # Determine device
        device = utils.get_available_devices(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing = True)
        self.sam2.eval()

        # Build Mask Generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=2,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            use_m2m=True,
            multimask_output=True,
        )  

        # Add Mask Filtering to Generator
        self.mask_generator = fmask.FilteredSAM2MaskGenerator(
            base_generator=self.mask_generator,
            min_area_filter=self.min_mask_area,
            max_rel_box_size=0.98,
        )

        # Initialize Domain Expert Classifier for Filtering False Positives
        if classifier:
            self.classifier = classifier
            self.target_class = target_class
            # Also set classifier to eval mode
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
        else:
            self.classifier = None
            self.target_class = None

    @torch.inference_mode()
    def segment_image(self,
        image0,
        display_image: bool = True,
        use_sliding_window: bool = True
    ):
        """
        Segment image using sliding window approach
        
        Args:
            image0: Input image
            display_image: Whether to display the result
            use_sliding_window: Whether to use sliding window (True) or single inference (False)
        """


        # Fourier Crop the Image to the Desired Resolution
        (nx, ny) = image0.shape
        if not use_sliding_window and (nx > 1536 or ny > 1536):
            scale_factor =  max(nx, ny) / 1024 
            image0 = FourierRescale2D.run(image0, scale_factor)
            (nx, ny) = image0.shape

        # Increase Contrast of Image and Normalize the Image to [0,1]        
        image0 = utils.contrast(image0, std_cutoff=2)
        image0 = utils.normalize(image0, rgb = False)

        # Extend From Grayscale to RGB 
        image = np.repeat(image0[..., None], 3, axis=2)   

        # Run Segmentation
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
                window_masks = self.mask_generator.generate(window_image)
                
                # Transform masks back to full image coordinates
                for mask in window_masks:
                    
                    # Reset Full Mask
                    full_mask[:] = 0
                    full_mask[y1:y2, x1:x2] = mask['segmentation']
                    
                    # Update mask dictionary
                    mask['segmentation'] = full_mask.copy()
                    mask['bbox'][0] += x1  # x offset
                    mask['bbox'][1] += y1  # y offset

                # Filter Out Small Masks and Add to All Masks
                window_masks = [mask for mask in window_masks if mask['area'] >= self.min_mask_area]
                all_masks.extend(window_masks)

            # Store the Masks
            self.masks = all_masks       
            
        else:
            # Original single inference
            self.masks = self.mask_generator.generate(image)
            
            # Filter Out Small Masks
            self.masks = [mask for mask in self.masks if mask['area'] >= self.min_mask_area]
            self.masks = sorted(self.masks, key=lambda mask: mask['area'], reverse=False)

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is not None:
            self.masks = filters.apply_classifier(image, self.masks, self.classifier, self.target_class)

        # Store image
        self.image = image0

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display_image:
            viz.display_mask(self.image, self.masks)

        # Return the Masks
        return self.masks  
        
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

    def masks_to_list(self, masks):
        if not masks:
            return None
    
        # Get shape from first mask
        h, w = masks[0]['segmentation'].shape
        n_masks = len(masks)
        
        # Create array
        masks_array = np.zeros((n_masks, h, w), dtype=np.uint8)
        for i, mask in enumerate(masks):
            masks_array[i] = mask['segmentation']
        
        return masks_array
        