import saber.analysis.estimate_thickness as estimate_thickness
from saber.visualization import classifier as viz
from saber.sam2 import tomogram_predictor
import saber.filters.masks as filters
from saber import pretrained_weights
from typing import List, Tuple
from saber.utils import io
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

# Suppress SAM2 Logger 
logger = logging.getLogger()
logger.disabled = True

class saber2Dsegmenter:
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
        self.device = io.get_available_devices(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=self.device, apply_postprocessing = True)
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

        # Initialize Image and Masks
        self.image = None

    @torch.inference_mode()
    def segment_image(self,
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

        # Run Segmentation
        if use_sliding_window:

            # Create Full Mask
            full_mask = np.zeros(self.image.shape[:2], dtype=np.uint16)

            # Get sliding windows
            windows = self.get_sliding_windows(self.image.shape)
            
            # Process each window
            all_masks = []
            for i, (y1, x1, y2, x2) in tqdm(enumerate(windows), total=len(windows)):
                # Extract window
                window_image = self.image[y1:y2, x1:x2]
                
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
            with torch.no_grad():
                self.masks = self.mask_generator.generate(self.image)

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is not None:
            self.masks = filters.apply_classifier(self.image, self.masks, self.classifier, self.target_class)
        else: # Since Order Doesn't Matter, Sort by Area for Saber GUI. 
            self.masks = sorted(self.masks, key=lambda mask: mask['area'], reverse=False)

        # Filter Out Small Masks
        self.masks = [mask for mask in self.masks if mask['area'] >= self.min_mask_area]

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display_image:
            viz.display_mask_list(self.image, self.masks)

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
    
class saber3Dsegmenter(saber2Dsegmenter):
    def __init__(self,
        sam2_cfg: str, 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025
    ):  
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, min_rel_box_size)

        # Build Tomogram Predictor (VOS Optimized)
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.video_predictor = tomogram_predictor.TomogramSAM2Adapter(cfg, checkpoint, self.device)  

        # Initialize Inference State
        self.inference_state = None

        # Flag to Plot the Z-Slice Confidence Estimations
        self.confidence_debug = False
        
    @torch.inference_mode()
    def propagate_segementation(
        self,
        mask_shape: Tuple[int, int, int]
    ):
        """
        Propagate Segmentation in 3D with Video Predictor
        """

        # middle_frame = int( mask_shape[0] // 2 )
        start_frame = self.ann_frame_idx

        # Pull out Masks for Multiple Classes
        nMasks = len(self.masks )
        vol_mask = np.zeros( [mask_shape[0], mask_shape[1], mask_shape[2]], dtype=np.uint8)

        # run propagation throughout the video and collect the results in a dict
        video_segments1 = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame, reverse=False):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments1[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }      

        # run propagation throughout the video and collect the results in a dict
        video_segments2 = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame-1, reverse=True):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments2[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Merge Video Segments to Return for Visualization / Analysis    
        video_segments = video_segments1 | video_segments2   
        vol_mask = filters.segments_to_mask(video_segments, vol_mask, mask_shape, nMasks)

        return vol_mask, video_segments

    def filter_video_segments(self, video_segments, captured_scores, mask_shape):
        """
        Filter out masks with low confidence scores.
        """

        # Populate the Frame Scores Array
        for frame_idx, scores in captured_scores.items():
            if frame_idx is None:
                continue

            score_values = np.concatenate([s.flatten() for s in scores])

            # Store these score values in the corresponding row.
            # If there are fewer scores than the allocated length, the remaining values stay zero.
            self.frame_scores[frame_idx, ] = score_values

        # Determine the Range Along Z-Axis for Each Organelle
        self.mask_boundaries = estimate_thickness.fit_organelle_boundaries(self.frame_scores, plot=self.confidence_debug)

        # Now, filter the video_segments.
        # For each frame, if the score for the first mask is above the threshold, keep the segmentation;
        # otherwise, replace with an array of zeros (or background).
        nMasks = len(self.masks)
        filtered_video_segments = {}
        for frame_idx, seg_dict in video_segments.items():
            # Check the score for the first mask; adjust if needed.
            filtered_video_segments[frame_idx] = {}  # Initialize the dictionary for this frame
            for mask_idx in range(nMasks):
                if self.mask_boundaries[frame_idx, mask_idx] > 0.5:
                    filtered_video_segments[frame_idx][mask_idx+1] = seg_dict[mask_idx+1]
                else:
                    # For null frames, create an empty mask for given object id.
                    filtered_video_segments[frame_idx][mask_idx+1] = np.full(seg_dict[1].shape, False, dtype=bool)

        # Convert Video Segments into Mask
        nFrames = len(video_segments)
        masks = np.zeros([nFrames, mask_shape[1], mask_shape[2]], dtype=np.uint8)
        masks = filters.segments_to_mask(filtered_video_segments, masks, mask_shape, nMasks)

        return masks, filtered_video_segments
    
    