from saber.segmenters.base import saber3Dsegmenter
from saber.visualization import sam2 as viz
from saber.filters import masks as filters
from scipy import ndimage
import numpy as np
import torch

class generalSegmenter(saber3Dsegmenter):
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
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, min_rel_box_size)

        # Flag to Bound the Segmentation to the Tomogram
        self.bound_segmentation = True        


    @torch.inference_mode()
    def segment(
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
        if self.bound_segmentation:
            self.frame_scores = np.zeros([vol.shape[0], len(self.masks)])
            vol_masks, video_segments = self.filter_video_segments(video_segments, captured_scores, mask_shape)
        else: # Convert Video Segments to Masks (Without Filtering)
            vol_masks = filters.segments_to_mask(video_segments, vol_masks, mask_shape, len(self.masks))

        # Display Segmentations
        if show_segmentations:
            viz.display_video_segmentation(video_segments, self.inference_state)

        # Reset Inference State
        self.video_predictor.reset_state(self.inference_state)

        return vol_masks

    