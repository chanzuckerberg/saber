from saber.segmenters.base import saber3Dsegmenter
import torch

class generalSegmenter(saber3Dsegmenter):
    def __init__(self,
        sam2_cfg: str = 'base', 
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
    def segment_3d(
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
        if self.inference_state is None:
            self.inference_state = self.video_predictor.create_inference_state_from_tomogram(vol)

        # Set Masks - Right now this is external
        self.masks = masks

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if not show_segmentations:  save_mask = True
        else:                       save_mask = False

        # Set up score capture hook
        captured_scores, hook_handle = self._setup_score_capture_hook()

        # Get Dimensions
        nx = len(self.inference_state['images'])
        ny, nz = self.masks[0].shape[0], self.masks[0].shape[1]
        
        # Set annotation frame
        self.ann_frame_idx = ann_frame_idx if ann_frame_idx is not None else nx // 2
        
        # Add masks to predictor
        self._add_masks_to_predictor(self.masks, self.ann_frame_idx, ny)
        
        # Propagate and filter
        mask_shape = (nx, ny, nz)
        vol_masks, video_segments = self._propagate_and_filter(
            vol, self.masks, captured_scores, mask_shape,
            filter_segmentation=self.bound_segmentation,
        )

        # Remove hook and Reset Inference State
        hook_handle.remove()
        self.video_predictor.reset_state(self.inference_state)

        return vol_masks