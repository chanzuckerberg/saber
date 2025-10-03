from saber.segmenters.general import generalSegmenter
from saber.utils import preprocessing
from tqdm import tqdm
import numpy as np
import torch


class fibSegmenter(generalSegmenter):

    def __init__(self, 
        sam2_cfg: str = 'base', 
        deviceID: int = 0, 
        classifier = None, 
        target_class: int = 1, 
        min_mask_area: int = 100, 
        min_rel_box_size: float = 0.025,
        ):
        """
        Initialize the fibSegmenter
        """
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, min_rel_box_size)
        self.ini_depth = 10 # Default spacing between slices to segment

    def segment(self, volume: np.ndarray, ini_depth: int, nframes: int = None):
        """
        Segment the volume

        Args:
            volume: The volume to segment
        """
        # Update ini_depth and nframes attributes
        self.ini_depth = ini_depth
        self.nframes = nframes

        # Segment the Volume
        if self.target_class > 0 or self.classifier is None:
            return self.single_segment(volume)
        else:
            return self.multiclass_segment(volume)

    @torch.inference_mode()
    def single_segment(self, volume: np.ndarray):
        """
        Segment the volume with a single class or without the classifier

        Args:
            volume: The volume to segment
            ini_depth: The spacing between slices to segment
        """

        # Single 3D array instead of accumulating 4D
        final_masks = np.zeros(volume.shape, dtype=np.uint16)

        # Main Loop
        for ii in tqdm(range(self.ini_depth, volume.shape[0], self.ini_depth)):

            # Set image and segment
            im = volume[ii]
            im = self._preprocess(im)            
            masks = self.segment_image(im, display_image=False)
            
            if len(masks) == 0:
                continue
                
            # Extract mask list
            mask_list = [m['segmentation'] for m in masks]
            
            # 3D propagation
            masks3d = self.segment_3d(volume, mask_list, ann_frame_idx=ii)
            
            # Convert to binary if target class specified
            if self.target_class > 0:
                masks3d = (masks3d > 0).astype(np.uint8)
            
            # Update final masks with maximum operation (in-place)
            np.maximum(final_masks, masks3d, out=final_masks)
        
        return final_masks

    @torch.inference_mode()
    def multiclass_segment(self, volume: np.ndarray):
        """
        Segment the volume with multiple classes using the classifier

        Args:
            volume: The volume to segment
        """
        # Instead of 4D array, use 3D for current best class and confidence
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        max_confidence = np.zeros(volume.shape, dtype=np.float32)

        # Main Loop
        for ii in tqdm(range(self.ini_depth, volume.shape[0], self.ini_depth)):

            # Call mask generator directly
            im = volume[ii]
            im = self._preprocess(im)
            raw_masks = self.mask_generator.generate(im)
            
            # Filter small masks
            raw_masks = [mask for mask in raw_masks if mask['area'] >= self.min_mask_area]
            
            if len(raw_masks) == 0:
                continue
            
            # Get classifier predictions
            mask_arrays = np.array([m['segmentation'].astype(np.uint8) for m in raw_masks])
            predictions = self.classifier.batch_predict(im[:,:,0], mask_arrays, self.batchsize)
            
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Only process non-background masks
            valid_indices = predicted_classes > 0
            if not np.any(valid_indices):
                continue
            
            # Prepare masks for 3D propagation
            mask_list = [raw_masks[i]['segmentation'] for i, valid in enumerate(valid_indices) if valid]
            valid_predictions = predictions[valid_indices]
            valid_classes = predicted_classes[valid_indices]
            
            # 3D propagation: Call the parent's segment through the wrapper
            masks3d = self.segment_3d(volume, mask_list, ann_frame_idx=ii)

            # Calculate frame range

            # Update with maximum confidence approach
            for idx, (probs, class_id) in enumerate(zip(valid_predictions, valid_classes)):
                mask_region = (masks3d == (idx + 1))

                if np.any(mask_region):
                    confidence = probs[class_id]
                    update_mask = mask_region & (confidence > max_confidence)
                    final_masks[update_mask] = class_id
                    max_confidence[update_mask] = confidence
        
        return final_masks