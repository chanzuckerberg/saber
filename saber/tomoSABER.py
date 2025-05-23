# Silence SAM2 loggers
import logging
logging.getLogger().setLevel(logging.WARNING)  # Root logger - blocks all INFO messages
# logging.getLogger("sam2").setLevel(logging.ERROR)  # Only show errors

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

class cryoTomoSegmenter:
    def __init__(self,
        sam2_cfg: str, 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.01
    ):  
        """
        Initialize the cryoTomoSegmenter
        """ 

        # Minimum Mask Area and Relative Box Size to Ignore 
        self.min_mask_area = min_mask_area
        self.min_rel_box_size = min_rel_box_size
        j
        # Determine device
        device = utils.determine_device(deviceID)

        # Build SAM2 model
        (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
        self.sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing = True)

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

        # Flag to indicate we're processing a 3D tomogram rather than a 2D image
        self.is_tomogram_mode = False

        # Initialize Inference State
        self.inference_state = None

    def segment_image(self,
        vol,
        slab_thickness: int,
        display_image: bool = True, 
        save_run: str = None,
        run = None, 
        zSlice: int = None,
    ):
        """
        Segment a 2D image using the Mask Generator
        """

        # Normalize the Volume from [0,1]
        vol = utils.normalize(vol)

        # Initialize Video Predictor
        if self.is_tomogram_mode and self.inference_state is None:
            self.inference_state = self.video_predictor.create_inference_state_from_tomogram(vol)

        # Project Tomogram to 2D Image
        if zSlice is None:
            zSlice = int(vol.shape[0] // 2)

        # Option 1: Project Multiple Slabs to Provide Z-Context
        # image1 = utils.project_tomogram(vol, zSlice - slab_thickness/3, slab_thickness)
        # image2 = utils.project_tomogram(vol, zSlice, slab_thickness)
        # image3 = utils.project_tomogram(vol, zSlice + slab_thickness/3, slab_thickness)

        # # # Extend From Grayscale to RGB 
        # image = np.stack([image1, image2, image3], axis=-1)
        # image = utils.contrast(image, std_cutoff=3)
        # # Normalize the Image to [0,1]        
        # image = utils.normalize(image, rgb = True)

        # # Hold Onto Original Image for Training
        # self.image = image

        # Option 2: Project Single Slab 
        image0 = utils.project_tomogram(vol, zSlice, slab_thickness)
        image0 = utils.contrast(image0, std_cutoff=3)
        image0 = utils.normalize(image0)
        image = np.stack([image0, image0, image0], axis=-1)

        # Hold Onto Original Image for Training
        self.image = image0

        # Run Inference from Pre-trained Model
        self.masks = self.mask_generator.generate(image)
        
        # Display Original SAM2 Segmentations
        plt.imshow(image, cmap='gray'); viz.show_anns(self.masks); plt.axis('off');  plt.show()

        # Apply Classifier Model or Physical Constraints to Filter False Positives
        if self.classifier is not None:
            self.masks = filters.apply_classifier(image, self.masks, self.classifier, self.target_class)

        # Filter Out Small Masks
        self.masks = [mask for mask in self.masks if mask['area'] >= self.min_mask_area]     

        # Debug: Display The Classified SAM2 Segmentations
        # plt.imshow(image, cmap='gray'); viz.show_anns(self.masks); plt.axis('off');  plt.show()
        # plt.imshow(np.sum([mask['segmentation'] for mask in masks], axis=0), cmap='hot'); plt.colorbar(); plt.show()
        # exit()

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if display_image:
            cryoviz.save_mask_segmentation(run, image, self.masks, save_run)

        return self.masks 

    def segment_tomogram(
        self, 
        vol,
        run,
        slab_thickness: int,
        show_segmentations: bool = False, 
        save_run: str = None, 
        zSlice: int = None
    ):  
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Testing Try 1D Smoothing along Z-Dimension
        vol = gauss.gaussian_smoothing(vol, 5, dim=0)

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if not show_segmentations:  save_mask = True
        else:                       save_mask = False
        self.is_tomogram_mode = True

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

        # Create Initial Masks 
        self.segment_image(
            vol, slab_thickness, display_image = save_mask, 
            save_run = save_run, run = run, zSlice = zSlice)

        # Check to Make Sure Masks are Found
        if len(self.masks) == 0:
            hook_handle.remove()
            return None

        # Get the dimensions of the volume.
        (nx, ny, nz) = (
            len(self.inference_state['images']),
            self.masks[0]['segmentation'].shape[0],
            self.masks[0]['segmentation'].shape[1]
        )

        prompts = {}
        # the frame index we interact with
        if zSlice is None:
            self.ann_frame_idx = int( nx // 2) 
        else:
            self.ann_frame_idx = int( zSlice )

        # Extract centers of mass for each mask (for prompting).
        auto_points = np.array([
            ndimage.center_of_mass(item['segmentation'])
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
                mask=self.masks[ii]["segmentation"],
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
        # estimate_thickness.save_frame_scores(run, self.frame_scores) # (Save Frame Scores )

        # Convert Video Segments to Masks (Without Filtering)
        # vol_masks = utils.convert_segments_to_mask(video_segments, vol_masks, mask_shape, len(self.masks))

        # Filter out low confidence masks at edges of tomograms
        if show_segmentations:
            viz.display_video_segmentation(video_segments, self.inference_state)

        # print(f'Saving Video Segmentation...')
        # utils.record_video_segmentation(
        #     video_segments, 
        #     self.inference_state,
        #     output_file = f'video_segmentation.mp4',
        #     fps = 10
        # )

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
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame, reverse=False):

            # Update current frame
            self.current_frame = out_frame_idx
            video_segments1[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
        vol_mask = utils.convert_segments_to_mask(video_segments1, vol_mask, mask_shape, nMasks)               

        # run propagation throughout the video and collect the results in a dict
        video_segments2 = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx= start_frame-1, reverse=True):

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