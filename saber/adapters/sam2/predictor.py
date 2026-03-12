from saber.adapters.base import BaseAdapter, SAM2AdapterConfig
from saber.adapters.preprocessing import TomogramPreprocessor
from typing import Optional, Tuple, Any, Dict, Iterator, List
from sam2.build_sam import build_sam2_video_predictor
from saber.adapters.sam2.automask import build_amg
from saber.adapters.sam2.amg import cfgAMG
from saber import pretrained_weights
from collections import OrderedDict
import skimage.transform
import numpy as np
import torch


class SAM2Adapter(BaseAdapter):
    """
    MIT Licensed adapter that provides a clean interface between tomogram data and SAM2.
    Implements BaseAdapter with hook-based presence score capture in segment_volume().
    """

    def __init__(self, config: SAM2AdapterConfig, device: str = "cuda"):
        cfg_path, checkpoint = pretrained_weights.get_sam2_checkpoint(config.cfg)

        self.predictor = build_sam2_video_predictor(
            cfg_path, checkpoint, device=device, vos_optimized=False,
        )

        if config.num_maskmem > 7:
            raise ValueError("num_maskmem must be less than 7")

        maskmem = self.predictor.maskmem_tpos_enc[:config.num_maskmem]
        self.predictor.maskmem_tpos_enc = torch.nn.Parameter(maskmem)
        if hasattr(self.predictor, 'num_maskmem'):
            self.predictor.num_maskmem = config.num_maskmem

        self.preprocessor = TomogramPreprocessor(config.light_modality)
        self.frame_metrics: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._vol_shape: Optional[Tuple[int, int, int]] = None
        self.inference_state = None
        self._current_frame = None
        self._config = config
        self._mask_generator = None

    # ------------------------------------------------------------------
    # 2D segmentation
    # ------------------------------------------------------------------

    def segment_image_2d(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run AMG-based 2D segmentation. Lazily builds the mask generator."""

        # Preprocess Image if it is 2D
        if image.ndim == 2:
            image = self.preprocessor.prepare(image, to_rgb=True)

        # Build Mask Generator if it is not already built
        if self._mask_generator is None:
            if self._config.amg_cfg is not None:
                amg_dict = self._config.amg_cfg.dict()
            else:
                amg_dict = cfgAMG(sam2_cfg=self._config.cfg).dict()
            self._mask_generator = build_amg(
                amg_dict, self._config.min_mask_area, device=self.predictor.device
            )
        return self._mask_generator.generate(image)

    # ------------------------------------------------------------------
    # Volume setter
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_volume(self, tomogram: np.ndarray, offload_video_to_cpu: bool = False) -> None:
        """Load a tomogram and prepare the SAM2 inference state."""
        self._vol_shape = tomogram.shape
        self.frame_metrics = {}
        self.inference_state = self.create_inference_state_from_tomogram(
            tomogram, offload_video_to_cpu=offload_video_to_cpu
        )

    # ------------------------------------------------------------------
    # State building
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def create_inference_state_from_tomogram(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        """Create inference state from tomogram data."""
        normalized_tomogram = self.preprocessor.normalize_tomogram(tomogram)

        images, video_height, video_width = self.preprocessor.load_grayscale_image_array(
            normalized_tomogram,
            image_size=self.predictor.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=self.predictor.device,
        )

        inference_state = self._create_empty_inference_state(
            images=images,
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    @torch.inference_mode()
    def _create_empty_inference_state(
        self,
        images: torch.Tensor,
        video_height: int,
        video_width: int,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
    ) -> Dict[str, Any]:
        """Create an empty inference state structure compatible with SAM2."""
        compute_device = self.predictor.device

        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
        }

        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def add_new_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray,
                     inference_state=None) -> Tuple:
        """Add new mask — delegates to SAM2 predictor."""
        state = inference_state or self.inference_state
        return self.predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

    def add_new_points_or_box(self, frame_idx: int, obj_id: int,
                               inference_state=None, **kwargs) -> Tuple:
        """Add new points or box — delegates to SAM2 predictor."""
        state = inference_state or self.inference_state
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            **kwargs,
        )

    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
        inference_state=None,
    ) -> Iterator:
        """
        Propagate tracking in video.
        Yields 5-tuple (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores=None)
        to match the SAM3 adapter interface.
        """
        state = inference_state or self.inference_state
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            yield frame_idx, obj_ids, mask_logits, mask_logits, None

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_masks(masks) -> List[np.ndarray]:
        """
        Accept masks in any format and return a list of (H, W) float32 arrays.
        Handles: None, torch.Tensor (N,1,H,W) or (N,H,W), list of (H,W) arrays/tensors.
        """
        if masks is None:
            return []

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if isinstance(masks, np.ndarray) and masks.ndim >= 3:
            return [np.squeeze(masks[i]).astype(np.float32) for i in range(masks.shape[0])]

        out = []
        for m in masks:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            if isinstance(m, dict):
                m = m['segmentation']
            out.append(np.squeeze(m).astype(np.float32))
        return out

    @torch.inference_mode()
    def segment_volume(
        self,
        start_frame_idx: int,
        masks=None,
        vol_shape=None,
        max_frame_num_to_track=None,
        min_presence_score: float = 0.5,
        inference_state=None,
    ) -> np.ndarray:
        """
        Bidirectional propagation with hook-based score capture.

        Populates self.frame_metrics[frame_idx][obj_id]["presence_score"] using
        estimate_thickness.fit_organelle_boundaries() on the hook-captured logits.

        Returns (Z, H, W) uint16 array.
        """
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before segment_volume().")

        if vol_shape is None:
            vol_shape = self._vol_shape
        if vol_shape is None:
            raise RuntimeError("vol_shape required when inference_state is passed explicitly.")

        Z, H, W = vol_shape

        # Seed from external masks
        mask_list = self._normalize_masks(masks)
        for obj_id, mask in enumerate(mask_list, start=1):
            if np.max(mask) == 0:
                continue
            self.add_new_mask(
                frame_idx=start_frame_idx,
                obj_id=obj_id,
                mask=mask,
                inference_state=state,
            )

        # Set up hook to capture object score logits from mask decoder
        self._current_frame = None
        captured_scores: Dict[Any, list] = {}

        def _hook(module, inputs, output):
            logits = output[3].detach().cpu().to(torch.float32).numpy()
            fidx = self._current_frame
            if fidx not in captured_scores:
                captured_scores[fidx] = []
            captured_scores[fidx].append(logits)

        hook_handle = self.predictor.sam_mask_decoder.register_forward_hook(_hook)
        self.frame_metrics = {}

        vol_masks = np.zeros((Z, H, W), dtype=np.uint16)

        def _apply(frame_idx, obj_ids, mask_logits):
            for i, obj_id in enumerate(obj_ids):
                m = mask_logits[i] > 0.0
                if hasattr(m, 'cpu'):
                    m = m.cpu().numpy()
                m = np.squeeze(m).astype(bool)
                if m.shape != (H, W):
                    m = skimage.transform.resize(m, (H, W), order=0, anti_aliasing=False)
                vol_masks[frame_idx] = np.where(m, int(obj_id), vol_masks[frame_idx])

        # Forward pass
        for frame_idx, obj_ids, mask_logits, _, _ in self.propagate_in_video(
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            inference_state=state,
        ):
            self._current_frame = frame_idx
            _apply(frame_idx, obj_ids, mask_logits)

        # Backward pass (only fills slices not already set by forward)
        for frame_idx, obj_ids, mask_logits, _, _ in self.propagate_in_video(
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=True,
            inference_state=state,
        ):
            self._current_frame = frame_idx
            if not vol_masks[frame_idx].any():
                _apply(frame_idx, obj_ids, mask_logits)

        hook_handle.remove()

        # Compute presence scores using hook-captured logits + fit_organelle_boundaries
        nMasks = len(mask_list)
        if nMasks > 0:
            import saber.filters.estimate_thickness as estimate_thickness

            frame_scores = np.zeros([Z, nMasks])
            for fidx, scores in captured_scores.items():
                if fidx is None:
                    continue
                score_values = np.concatenate([s.flatten() for s in scores])
                n = min(len(score_values), nMasks)
                frame_scores[fidx, :n] = score_values[:n]

            mask_boundaries = estimate_thickness.fit_organelle_boundaries(
                frame_scores, plot=False
            )

            for fidx in range(Z):
                self.frame_metrics[fidx] = {}
                for mask_idx in range(nMasks):
                    obj_id = mask_idx + 1
                    presence_score = float(mask_boundaries[fidx, mask_idx])
                    self.frame_metrics[fidx][obj_id] = {"presence_score": presence_score}
                    if presence_score < min_presence_score:
                        vol_masks[fidx][vol_masks[fidx] == obj_id] = 0

        return vol_masks.astype(np.uint16)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, inference_state=None) -> None:
        """Reset state — delegates to SAM2 predictor."""
        state = inference_state or self.inference_state
        if state is not None:
            self.predictor.reset_state(state)

    def clear_all_prompts_in_frame(self, *args, **kwargs):
        """Clear prompts — delegates to SAM2 predictor."""
        return self.predictor.clear_all_prompts_in_frame(*args, **kwargs)

    def remove_object(self, *args, **kwargs):
        """Remove object — delegates to SAM2 predictor."""
        return self.predictor.remove_object(*args, **kwargs)
