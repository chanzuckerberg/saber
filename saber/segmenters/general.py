# Silence SAM2 loggers
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

from saber.visualization import cryosam2 as cryoviz, sam2 as viz
import saber.process.estimate_thickness as estimate_thickness
import saber.process.mask_filters as filters
import torch, skimage, os, cv2, saber
from saber import pretrained_weights
from typing import Tuple, Optional
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
from saber.sam2 import tomogram_predictor, filtered_automatic_mask_generator as fmask
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

class generalSegmenter:
    def __init__(self,
        sam2_cfg: str, 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025
    ):  
        """
        Initialize the generalSegmenter
        """ 

        # Minimum Mask Area and Relative Box Size to Ignore 
        self.min_mask_area = min_mask_area
        self.min_rel_box_size = min_rel_box_size
        
        # Determine device
        device = utils.determine_device(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing = True)
        self.sam2.eval()

        # Build Default Mask Generator
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
            min_rel_box_size=self.min_rel_box_size,
            min_area_filter=self.min_mask_area,
        )

        # Build Tomogram Predictor (VOS Optimized)
        self.video_predictor = tomogram_predictor.TomogramSAM2Adapter(cfg, checkpoint, device)     

        # Initialize Domain Expert Classifier for Filtering False Positives
        if classifier:
            self.classifier = classifier
            self.target_class = target_class
        else:
            self.classifier = None
            self.target_class = None

        # Initialize Inference State
        self.inference_state = None

    @torch.inference_mode()
    def segment_volume(
        self,
        vol,
        masks,
        ann_frame_idx: int = None,
        show_segmentations: bool = False
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Create Inference State
        self.inference_state = self.video_predictor.create_inference_state_from_tomogram(vol)

        # Set Masks - Right now this is external
        self.masks = masks

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if not show_segmentations:  save_mask = True
        else:                       save_mask = False

        # Set up a dictionary to capture the object score logits from the mask decoder.
        # The keys will be frame indices and the values will be a list of score arrays from that frame.
        captured_scores = {}

        # We'll use an attribute to store the current frame index. It will be updated in propagate_segementation.
        self.current_frame = None

        # Define a Hook to Capture the Object Score Logits
        def mask_decoder_hook(module, inputs, output):
            """
            This hook captures the object score logits every time the SAM mask decoder is run.
            The expected output tuple is: (low_res_multimasks, ious, sam_output_tokens, object_score_logits)
            Since IoUs aren't provided in your version, we capture the object score logits (element index 3).
            """
            # Convert logits from bfloat16 to float32 before converting to NumPy.
            logits = output[3].detach().cpu().to(torch.float32).numpy()

            frame_idx = self.current_frame
            if frame_idx not in captured_scores:
                captured_scores[frame_idx] = []
            captured_scores[frame_idx].append(logits)

        # Register the hook on the SAM mask decoder.
        hook_handle = self.video_predictor.predictor.sam_mask_decoder.register_forward_hook(mask_decoder_hook)
            
        # Check to Make Sure Masks are Found
        if len(self.masks) == 0:
            hook_handle.remove()
            return None

        # Get the dimensions of the volume.
        (nx, ny, nz) = (
            len(self.inference_state['images']),
            self.masks[0].shape[0],
            self.masks[0].shape[1]
        )

        prompts = {}
        # the frame index we interact with
        if ann_frame_idx is None:
            self.ann_frame_idx = int( nx // 2) 
        else:
            self.ann_frame_idx = int( ann_frame_idx )

        # Extract centers of mass for each mask (for prompting).
        auto_points = np.array([
            ndimage.center_of_mass(item)
            for item in self.masks ])[:, ::-1]

        # Map Segmentation back to full resolution and positive points for segmentation
        scale = self.video_predictor.predictor.image_size / ny
        labels = np.array([1], np.int32)
        for ii in range(auto_points.shape[0]):

            sam_points = ( auto_points[ii,:] * scale ).reshape(1, 2)
            ann_obj_id = ii + 1 # give a unique id to each object we interact with (it can be any integers)

            # Predict with Masks
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                mask=self.masks[ii],
            )

            prompts.setdefault(ann_obj_id, {})
            prompts[ann_obj_id].setdefault(self.ann_frame_idx, [])
            prompts[ann_obj_id][self.ann_frame_idx].append((sam_points, labels)) 

        # Propagate Segmentation in 3D
        mask_shape = (nx, ny, nz)
        vol_masks, video_segments = self.propagate_segementation( mask_shape )
        hook_handle.remove()

        # Filter out low confidence masks at edges of tomograms
        self.frame_scores = np.zeros([vol.shape[0], len(self.masks)])
        vol_masks, video_segments = self.filter_video_segments(video_segments, captured_scores, mask_shape)

        # Filter out low confidence masks at edges of tomograms
        if show_segmentations:
            viz.display_video_segmentation(video_segments, self.inference_state)

        # Reset Inference State
        self.video_predictor.reset_state(self.inference_state)

        return vol_masks

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
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame + 1, reverse=False):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments1[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
        vol_mask = utils.convert_segments_to_mask(video_segments1, vol_mask, mask_shape, nMasks)               

        # run propagation throughout the video and collect the results in a dict
        video_segments2 = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame, reverse=True):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments2[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
        vol_mask = utils.convert_segments_to_mask(video_segments2, vol_mask, mask_shape, nMasks)

        # Merge Video Segments to Return for Visualization / Analysis    
        video_segments = video_segments1 | video_segments2   

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
        self.mask_boundaries = estimate_thickness.fit_organelle_boundaries(self.frame_scores)
        
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
        masks = utils.convert_segments_to_mask(filtered_video_segments, masks, mask_shape, nMasks)

        return masks, filtered_video_segments