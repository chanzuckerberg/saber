from saber.segmenters.base import saber3D
from saber.utils import preprocessing
from saber.segmenters import utils
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch

class propagationSegmenter(saber3D):

    def __init__(self,
        deviceID: int = 0,
        cfg: Optional[AdapterConfig] = None,
        amg_cfg: Optional[cfgAMG] = None,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025,
    ):
        """
        Segmenter that combines 2D mask generation with SAM2 3D propagation.

        For each seed slice, generates 2D masks (via classifier or text prompt),
        then propagates them through the volume using the video predictor.

        Args:
            deviceID: GPU device index.
            cfg: Adapter config (SAM2AdapterConfig or SAM3AdapterConfig).
            amg_cfg: Optional AMG hyperparameters for mask generation.
            min_mask_area: Minimum pixel area for a mask to be kept.
            min_rel_box_size: Minimum bounding-box size relative to image width.
        """
        self.min_rel_box_size = min_rel_box_size
        super().__init__(
            deviceID=deviceID, cfg=cfg, amg_cfg=amg_cfg,
            min_mask_area=min_mask_area,
        )
        self.ini_depth = 10

    @torch.inference_mode()
    def segment_3d(self, vol, masks, ann_frame_idx: int = None):
        """
        Propagate 2D seed masks through a 3D volume.

        Args:
            vol: Volume array of shape (nx, ny, nz).
            masks: List of 2D boolean mask arrays to use as seeds.
            ann_frame_idx: Slice index where seeds were generated.
                           Defaults to the middle slice if not provided.
            display: Unused; reserved for future visualization support.

        Returns:
            3D mask array of shape (nx, ny, nz) with propagated instance labels.
        """
        if not self._vol_loaded:
            self.video_predictor.set_volume(vol)
            self._vol_loaded = True

        self.masks = masks
        nx = vol.shape[0]
        ny, nz = self.masks[0].shape[0], self.masks[0].shape[1]
        self.ann_frame_idx = ann_frame_idx if ann_frame_idx is not None else nx // 2
        return self.propagate((nx, ny, nz))

    def segment(self, volume: np.ndarray, ini_depth: int, nframes: int = None, target_class: int = 1, text_prompt: str = None, display: bool = False):
        """
        Top-level entry point for segmenting a volume.

        Routes to single_segment (one target class or text prompt) or
        multiclass_segment (all classes via classifier, target_class=0).

        Args:
            volume: 3D array of shape (nx, ny, nz).
            ini_depth: Stride between seed slices.
            nframes: Number of frames to propagate around each seed slice.
            target_class: Class index to keep. 0 retains all non-background classes
                          (multiclass mode, requires a classifier).
            text_prompt: Text description used by the SAM3 adapter instead of a classifier.

        Returns:
            Segmentation mask array of shape (nx, ny, nz).
        """
        self.ini_depth = ini_depth
        self.nframes = nframes
        self.target_class = target_class
        self.display = display
        if self.target_class > 0 or self.classifier is None:
            return self.single_segment(volume, text_prompt=text_prompt)
        else:
            return self.multiclass_segment(volume)

    @torch.inference_mode()
    def single_segment(self, volume: np.ndarray, text_prompt: str = None):
        """
        Segment a volume targeting a single class (or any foreground via text prompt).

        For each seed slice, runs segment_image to get 2D masks filtered to
        target_class, then propagates them through the volume with segment_3d.
        Results from all seed slices are merged via element-wise max.

        Args:
            volume: 3D array of shape (nx, ny, nz).
            text_prompt: Forwarded to segment_image for SAM3-based segmentation.

        Returns:
            Separated instance mask array of shape (nx, ny, nz).
        """
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        for ii in tqdm(range(2, volume.shape[0], self.ini_depth)):
            masks = self.segment_image(volume[ii], display=False, target_class=self.target_class, text_prompt=text_prompt)
            if len(masks) == 0:
                continue
            mask_list = [m['segmentation'] for m in masks]
            masks3d = self.segment_3d(volume, mask_list, ann_frame_idx=ii)
            if self.target_class > 0:
                masks3d = (masks3d > 0).astype(np.uint8)
            np.maximum(final_masks, masks3d, out=final_masks)
        return utils.separate_masks(final_masks)

    @torch.inference_mode()
    def multiclass_segment(self, volume: np.ndarray):
        """
        Segment a volume retaining all predicted classes from the classifier.

        For each seed slice, generates raw 2D masks via the adapter, runs the
        classifier to predict class labels, keeps all non-background masks, and
        propagates them with segment_3d. Voxels are assigned the class with the
        highest confidence across all seed slices.

        Args:
            volume: 3D array of shape (nx, ny, nz).

        Returns:
            Multiclass mask array of shape (nx, ny, nz) with class IDs as values.
        """
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        max_confidence = np.zeros(volume.shape, dtype=np.float32)
        for ii in tqdm(range(2, volume.shape[0], self.ini_depth)):
            im = preprocessing.prepare(volume[ii], to_rgb=True)
            raw_masks = self.adapter.segment_image_2d(im, target_class=self.target_class)
            raw_masks = [mask for mask in raw_masks if mask['area'] >= self.min_mask_area]
            if len(raw_masks) == 0:
                continue
            mask_arrays = np.array([m['segmentation'].astype(np.uint8) for m in raw_masks])
            predictions = self.classifier.batch_predict(im[:,:,0], mask_arrays, self.batchsize)
            predicted_classes = np.argmax(predictions, axis=1)
            valid_indices = predicted_classes > 0
            if not np.any(valid_indices):
                continue
            mask_list = [raw_masks[i]['segmentation'] for i, valid in enumerate(valid_indices) if valid]
            valid_predictions = predictions[valid_indices]
            valid_classes = predicted_classes[valid_indices]
            masks3d = self.segment_3d(volume, mask_list, ann_frame_idx=ii)
            for idx, (probs, class_id) in enumerate(zip(valid_predictions, valid_classes)):
                mask_region = (masks3d == (idx + 1))
                if np.any(mask_region):
                    confidence = probs[class_id]
                    update_mask = mask_region & (confidence > max_confidence)
                    final_masks[update_mask] = class_id
                    max_confidence[update_mask] = confidence
        return final_masks

    @torch.inference_mode()
    def slice_by_slice(self, volume: np.ndarray, text_prompt: str):
        """
        Segment a volume by applying 2D segmentation to each slice independently.

        This is a fallback method that does not use any temporal propagation.
        It applies segment_image to each slice and merges results via element-wise max.

        Args:
            volume: 3D array of shape (nx, ny, nz).
            text_prompt: Forwarded to segment_image for SAM3-based segmentation.

        Returns:
            Separated instance mask array of shape (nx, ny, nz).
        """
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        masks3d = np.zeros(volume.shape[1:], dtype=np.uint16)
        for ii in tqdm(range(volume.shape[0])):
            masks = self.segment_image(volume[ii], display=False, text_prompt=text_prompt)
            if len(masks) == 0:
                continue
            mask_list = [m['segmentation'] for m in masks]
            for idx, mask in enumerate(mask_list):
                masks3d[mask] = idx + 1
            np.maximum(final_masks[ii], masks3d, out=final_masks[ii])
            masks3d[:] = 0
        return utils.separate_masks(final_masks)
