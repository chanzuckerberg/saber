import saber.filters.estimate_thickness as estimate_thickness
from saber.visualization import classifier as viz
import saber.visualization.sam2 as vidviz
from saber.sam2 import tomogram_predictor
from saber.utils import preprocessing
import saber.filters.masks as filters
from saber import pretrained_weights
from saber.segmenters import utils
from typing import List, Tuple
from saber.utils import io
from scipy import ndimage
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
        sam2_cfg: str = 'base', 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 50,
        window_size: int = 256,
        overlap_ratio: float = 0.25
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
        self.deviceID = deviceID

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=self.device, apply_postprocessing = True)
        self.sam2.eval()

        # Build Mask Generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,             # 16
            points_per_batch=64,            # 128
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=2,                # 1
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
            self.batchsize = 32
            # Also set classifier to eval mode
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
        else:
            self.classifier = None
            self.target_class = None
            self.batchsize = None

        # Initialize Image and Masks
        self.image = None

        # Internal Variable to Let Users Save Segmentations 
        self.save_button = False
        self.remove_repeating_masks = False        

    @torch.inference_mode()
    def segment_image(self,
        image: np.ndarray,
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

        # Preprocess image if it is 2D
        if image.ndim == 2:
            image = self._preprocess(image)

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

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is not None:
            self.masks = filters.apply_classifier(image, self.masks, self.classifier,
                                                  self.target_class, self.batchsize)
        else: # Since Order Doesn't Matter, Sort by Area for Saber GUI. 
            self.masks = sorted(self.masks, key=lambda mask: mask['area'], reverse=False)

        # Filter Out Small Masks and Duplicates
        self.masks = [mask for mask in self.masks if mask['area'] >= self.min_mask_area]
        if self.remove_repeating_masks:
            self.masks = utils.remove_duplicate_masks(self.masks)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display_image:
            viz.display_mask_list(image, self.masks, self.save_button)

        # Return the Masks
        self.image = image
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
    
    def _preprocess(self, image: np.ndarray):
        image = preprocessing.contrast(image, std_cutoff=3)
        image = preprocessing.normalize(image, rgb=False)
        image = np.repeat(image[..., None], 3, axis=2)
        return image
    
class saber3Dsegmenter(saber2Dsegmenter):
    def __init__(self,
        sam2_cfg: str = 'base', 
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

        # Minimum Logits Threshold for Confidence
        self.min_logits = 0.5        

        # Flag to Plot the Z-Slice Confidence Estimations
        self.confidence_debug = False

        # Default to full volume propagation
        self.nframes = None 
        
    @torch.inference_mode()
    def propagate_segementation(
        self,
        mask_shape: Tuple[int, int, int],
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
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            self.inference_state, start_frame_idx= start_frame, max_frame_num_to_track = self.nframes, reverse=False ):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments1[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > self.min_logits).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        # run propagation throughout the video and collect the results in a dict
        video_segments2 = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            self.inference_state, start_frame_idx= start_frame-1, max_frame_num_to_track = self.nframes, reverse=True ):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments2[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > self.min_logits).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Merge Video Segments to Return for Visualization / Analysis    
        video_segments = video_segments1 | video_segments2   
        vol_mask = filters.segments_to_mask(video_segments, vol_mask, mask_shape, nMasks)

        return vol_mask, video_segments

    def _setup_score_capture_hook(self):
        """
        Set up hook to capture object score logits from mask decoder.
        Returns: (captured_scores dict, hook_handle)
        """
        captured_scores = {}
        self.current_frame = None
        
        def mask_decoder_hook(module, inputs, output):
            """Capture object score logits from SAM mask decoder output."""
            logits = output[3].detach().cpu().to(torch.float32).numpy()
            frame_idx = self.current_frame
            if frame_idx not in captured_scores:
                captured_scores[frame_idx] = []
            captured_scores[frame_idx].append(logits)
        
        hook_handle = self.video_predictor.predictor.sam_mask_decoder.register_forward_hook(mask_decoder_hook)
        return captured_scores, hook_handle
    
    def _add_masks_to_predictor(self, masks, ann_frame_idx, ny):
        """
        Add masks to the video predictor with automatic prompting.
        
        Args:
            masks: List of mask arrays or mask dictionaries
            ann_frame_idx: Frame index for annotation
            ny: Height dimension for scaling
        
        Returns:
            prompts: Dictionary of prompts added
        """
        # Handle both mask arrays and mask dictionaries
        if isinstance(masks[0], dict):
            mask_arrays = [m['segmentation'] for m in masks]
        else:
            mask_arrays = masks
            
        # Extract centers of mass for prompting
        auto_points = np.array([
            ndimage.center_of_mass(mask)
            for mask in mask_arrays
        ])[:, ::-1]
        
        # Set up prompts
        prompts = {}
        scale = self.video_predictor.predictor.image_size / ny
        labels = np.array([1], np.int32)
        
        for ii, mask in enumerate(mask_arrays):
            sam_points = (auto_points[ii, :] * scale).reshape(1, 2)
            ann_obj_id = ii + 1
            
            # Add new mask
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask,
            )
            
            prompts.setdefault(ann_obj_id, {})
            prompts[ann_obj_id].setdefault(ann_frame_idx, [])
            prompts[ann_obj_id][ann_frame_idx].append((sam_points, labels))
            
        return prompts

    def _propagate_and_filter(self, vol, masks, captured_scores, mask_shape, 
                              filter_segmentation=True, show_segmentations=False):
        """
        Propagate segmentation and optionally filter results.
        
        Args:
            vol: Volume array
            masks: Input masks
            captured_scores: Captured confidence scores
            mask_shape: Shape of the output mask
            filter_segmentation: Whether to filter low-confidence segments
            show_segmentations: Whether to display results
            
        Returns:
            vol_masks: Final segmentation masks
            video_segments: Video segmentation dictionary
        """
        # Propagate segmentation
        vol_masks, video_segments = self.propagate_segementation(mask_shape)
        
        # Filter if requested
        if filter_segmentation:
            self.frame_scores = np.zeros([vol.shape[0], len(masks)])
            vol_masks, video_segments = self.filter_video_segments(
                video_segments, captured_scores, mask_shape
            )
        else:
            vol_masks = filters.segments_to_mask(
                video_segments, vol_masks, mask_shape, len(masks)
            )
        
        # Display if requested
        if show_segmentations:
            vidviz.display_video_segmentation(video_segments, self.inference_state)
            
        return vol_masks, video_segments        

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
        masks = np.zeros([mask_shape[0], mask_shape[1], mask_shape[2]], dtype=np.uint8)
        masks = filters.segments_to_mask(filtered_video_segments, masks, mask_shape, nMasks)

        return masks, filtered_video_segments
