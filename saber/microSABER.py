from typing import List, Dict, Any, Tuple, Optional
import saber.process.mask_filters as filters
from saber.visualization import sam2 as viz
import torch, skimage, os, cv2, saber
from saber import pretrained_weights
from scipy.optimize import curve_fit
import saber.utilities as utils
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

# Testing 1D Gaussian Kernel
from saber.process import gaussian_smooth as gauss

# Suppress Warning for Post Processing from SAM2 - 
# Explained Here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from sam2.build_sam import build_sam2, build_sam2_tomogram_predictor, build_sam2_video_predictor
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
        labeled_points: Optional[str] = None
    ):
        """
        Class for Segmenting Micrographs or Images using SAM2
        """

        # Minimum Mask Area to Ignore 
        self.min_mask_area = 100

        # Determine device
        device = utils.get_available_devices(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing = True)

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
            min_rel_box_size=0.1,
        )  

        # Initialize Domain Expert Classifier for Filtering False Positives
        if classifier:
            self.classifier = classifier
            self.target_class = target_class
        else:
            self.classifier = None
            self.target_class = None

        # Labeled Points to Overlay on Segmentations
        self.labeled_points = labeled_points

    def segment_image(self,
        image0,
        display_image: bool = True, 
    ):

        # Increase Contrast of Image and Normalize the Image to [0,1]        
        image0 = utils.contrast(image0, std_cutoff=2)
        image0 = utils.normalize(image0, rgb = False)

        # Extend From Grayscale to RGB 
        image = np.repeat(image0[..., None], 3, axis=2)   

        # Run Inference from Pre-trained SAM2 Model
        self.masks = self.mask_generator.generate(image)

        # Filter Out Small Masks
        self.masks = [mask for mask in self.masks if mask['area'] >= self.min_mask_area]

        # Debug: Display The SAM2 Segmentations
        # plt.imshow(image, cmap='gray'); viz.show_anns(self.masks); plt.axis('off'); plt.savefig('segmentations.png'); plt.show()
        # plt.imshow(np.sum([mask['segmentation'] for mask in masks], axis=0), cmap='hot'); plt.colorbar(); plt.show()
        # exit()

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is not None:
            self.masks = filters.apply_classifier(image, self.masks, self.classifier, self.target_class)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display_image:
            self.save_mask_segmentation(run, image, save_run)

        # Option 2: RGB Image
        self.image = image0

        # Return the Masks
        return self.masks     

    def save_mask_segmentation(self, masks):
        # TODO: Implement this
        pass