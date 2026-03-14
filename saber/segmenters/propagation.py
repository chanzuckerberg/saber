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
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025,
    ):
        self.target_class = target_class
        self.min_rel_box_size = min_rel_box_size
        super().__init__(
            deviceID=deviceID, cfg=cfg, amg_cfg=amg_cfg,
            min_mask_area=min_mask_area,
        )
        self.ini_depth = 10

    @torch.inference_mode()
    def segment_3d(self, vol, masks, ann_frame_idx: int = None, show_segmentations: bool = False):
        if not self._vol_loaded:
            self.video_predictor.set_volume(vol)
            self._vol_loaded = True

        self.masks = masks
        nx = vol.shape[0]
        ny, nz = self.masks[0].shape[0], self.masks[0].shape[1]
        self.ann_frame_idx = ann_frame_idx if ann_frame_idx is not None else nx // 2
        return self.propagate((nx, ny, nz))

    def segment(self, volume: np.ndarray, ini_depth: int, nframes: int = None):
        self.ini_depth = ini_depth
        self.nframes = nframes
        if self.target_class > 0 or self.classifier is None:
            return self.single_segment(volume)
        else:
            return self.multiclass_segment(volume)

    @torch.inference_mode()
    def single_segment(self, volume: np.ndarray):
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        for ii in tqdm(range(2, volume.shape[0], self.ini_depth)):
            im = preprocessing.prepare(volume[ii], to_rgb=True)
            masks = self.segment_image(im, display=False)
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
        final_masks = np.zeros(volume.shape, dtype=np.uint16)
        max_confidence = np.zeros(volume.shape, dtype=np.float32)
        for ii in tqdm(range(2, volume.shape[0], self.ini_depth)):
            im = preprocessing.prepare(volume[ii], to_rgb=True)
            raw_masks = self.adapter.segment_image_2d(im)
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
