from saber.adapters.base import BaseAdapter, SAM3AdapterConfig
from saber.adapters.preprocessing import TomogramPreprocessor
from typing import Any, Dict, Iterator, List, Optional, Tuple
from saber.pretrained_weights import get_sam3_bpe_path, get_sam3_checkpoint
from sam3.model_builder import build_sam3_video_model
from saber.utils import preprocessing as prep
import numpy as np
import torch

_SAM3_WEIGHTS_MISSING_MSG = (
    "SAM3 weights not found. To download them:\n"
    "  1. Request access at https://huggingface.co/facebook/sam3\n"
    "  2. Log in:  huggingface-cli login\n"
)


def _sam3_output_to_mask_list(
    output: Dict[str, Any], min_mask_area: int ) -> List[Dict[str, Any]]:
    """Convert Sam3Processor output dict to AMG-compatible list of dicts."""
    masks_tensor = output.get("masks")
    scores = output.get("scores", None)
    if masks_tensor is None:
        return []
    if hasattr(masks_tensor, "cpu"):
        masks_np = masks_tensor.cpu().numpy()
    else:
        masks_np = np.array(masks_tensor)
    result = []
    for i in range(masks_np.shape[0]):
        mask = np.squeeze(masks_np[i]) > 0.5
        area = int(mask.sum())
        if area < min_mask_area:
            continue
        entry: Dict[str, Any] = {"segmentation": mask, "area": area}
        if scores is not None:
            s = scores[i]
            entry["predicted_iou"] = float(s.item() if hasattr(s, "item") else s)
        result.append(entry)
    return result


class SAM3Adapter(BaseAdapter):
    """
    Adapter for running SAM3 segmentation on tomogram data.

    Mirrors the SAM2Adapter interface so the two can be swapped.
    Uses SAM3's SAM2-compatible tracker API (Sam3TrackerPredictor) directly.

    Typical workflow
    ----------------
    >>> adapter = SAM3Adapter(SAM3AdapterConfig(), device="cuda")
    >>> adapter.set_volume(vol)                       # (Z, H, W) numpy array
    >>> adapter.add_new_mask(zSlice, obj_id=1, mask)  # binary (H, W) numpy array
    >>> vol_masks = adapter.segment_volume(zSlice)    # (Z, H, W) uint16
    """

    def __init__(
        self,
        config: SAM3AdapterConfig,
        device: str = "cuda",
    ):
        """
        Args:
            config: SAM3AdapterConfig with checkpoint and loading options.
            device: PyTorch device string (e.g. "cuda", "cuda:1").
        """

        self.preprocessor = TomogramPreprocessor(config.light_modality)
        self.device = torch.device(device)
        self._processor = None
        self.predictor = None
        self._config = config

        # Internal state — set by set_volume()
        self.inference_state: Optional[Dict[str, Any]] = None
        self._vol_shape: Optional[Tuple[int, int, int]] = None

        # Per-frame metrics populated by segment_volume().
        self.frame_metrics: Dict[int, Dict[int, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # 2D segmentation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def segment_image_2d(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Run text-prompted 2D segmentation. Lazily builds the Sam3Processor."""
        prompt = text_prompt or self._config.text_prompt
        if not prompt:
            raise ValueError("text_prompt required for SAM3 2D segmentation")

        # Build the Sam3Processor if it is not already built
        if self._processor is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            local_ckpt = get_sam3_checkpoint()
            ckpt_path = self._config.checkpoint_path or local_ckpt
            use_hf = self._config.load_from_HF and (ckpt_path is None)
            if ckpt_path is None and not use_hf:
                raise RuntimeError(_SAM3_WEIGHTS_MISSING_MSG)
            self._processor = Sam3Processor(
                build_sam3_image_model(
                    checkpoint_path=ckpt_path,
                    load_from_HF=use_hf,
                    bpe_path=get_sam3_bpe_path(),
                ),
                device=str(self.device),
            )

        image = prep.prepare(image) 
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        
        # Apply threshold to filter out low-confidence masks (if scores are available)
        keep = output['scores'] >= threshold
        output['masks'] = output['masks'][keep]
        output['masks_logits'] = output['masks_logits'][keep]
        output['scores'] = output['scores'][keep]

        # Convert to list of dicts with binary masks and metadata (area, predicted_iou)
        return _sam3_output_to_mask_list(output, self._config.min_mask_area)

    # ------------------------------------------------------------------
    # Volume setter
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_volume(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool = False,
    ) -> None:
        """
        Load a tomogram and prepare the SAM3 inference state.

        Call this once per volume.  After calling set_volume() you can add
        prompts and run segment_volume() without passing state explicitly.
        """

        # Build the SAM3 model if it is not already built
        if self.predictor is None:
            local_ckpt = get_sam3_checkpoint()
            ckpt_path = self._config.checkpoint_path or local_ckpt
            use_hf = self._config.load_from_HF and (ckpt_path is None)
            if ckpt_path is None and not use_hf:
                raise RuntimeError(_SAM3_WEIGHTS_MISSING_MSG)
            sam3_model = (
                build_sam3_video_model(
                    checkpoint_path=ckpt_path,
                    load_from_HF=use_hf,
                    bpe_path=get_sam3_bpe_path(),
                )
                .to(self.device)
                .eval()
            )

            # Use the SAM2-compatible tracker directly (as shown in the notebook)
            self.predictor = sam3_model.tracker
            self.predictor.backbone = sam3_model.detector.backbone
        
        # Set the volume shape and frame metrics
        self._vol_shape = tomogram.shape
        self.frame_metrics = {}
        self.inference_state = self._create_inference_state(
            tomogram, offload_video_to_cpu
        )

    # ------------------------------------------------------------------
    # Preprocessing / state building
    # ------------------------------------------------------------------

    def _preprocess_tomogram(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Normalize (Z, H, W) numpy array → float32 tensor
        (Z, 3, image_size, image_size) in [-1, 1].
        """
        t_min = float(tomogram.min())
        t_max = float(tomogram.max())
        if t_max > t_min:
            norm01 = (tomogram.astype(np.float32) - t_min) / (t_max - t_min)
        else:
            norm01 = np.zeros_like(tomogram, dtype=np.float32)

        images, orig_h, orig_w = self.preprocessor.load_grayscale_image_array(
            norm01,
            image_size=self.predictor.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=self.device,
        )
        return images, orig_h, orig_w

    @torch.inference_mode()
    def _create_inference_state(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a SAM3 tracker inference state from a tomogram array.
        """
        Z, H, W = tomogram.shape
        images, _orig_h, _orig_w = self._preprocess_tomogram(
            tomogram, offload_video_to_cpu
        )

        state = self.predictor.init_state(
            video_height=H,
            video_width=W,
            num_frames=Z,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        state["images"] = images
        return state

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def add_new_mask(
        self,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any, Any, Any]:
        """Seed segmentation on a Z-slice using a binary mask."""
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before add_new_mask().")
        mask_t = torch.as_tensor(mask, dtype=torch.float32)
        return self.predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask_t,
        )

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        frame_idx: int,
        obj_id: int,
        inference_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[int, Any, Any, Any]:
        """Seed segmentation using points or a bounding box."""
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before add_new_points_or_box().")
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            **kwargs,
        )

    @torch.inference_mode()
    def add_box_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        box_xyxy_norm: List[float],
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any, Any, Any]:
        """Seed segmentation on a Z-slice using a bounding box."""
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before add_box_prompt().")
        rel_box = np.array([box_xyxy_norm], dtype=np.float32)
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=rel_box,
        )

    @torch.inference_mode()
    def add_point_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        points_norm: np.ndarray,
        labels: np.ndarray,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any, Any, Any]:
        """Seed segmentation on a Z-slice using point clicks."""
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before add_point_prompt().")
        pts_t = torch.as_tensor(points_norm, dtype=torch.float32)
        lbl_t = torch.as_tensor(labels, dtype=torch.int32)
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=pts_t,
            labels=lbl_t,
        )

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[int, Any, Any, Any, Any]]:
        """
        Propagate segmentation through Z-slices in one direction.

        Yields:
            (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
        """
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before propagate_in_video().")
        yield from self.predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            propagate_preflight=True,
        )

    @staticmethod
    def _normalize_masks(masks) -> List[np.ndarray]:
        """
        Accept masks in any of these formats and return a list of (H, W) arrays:

        - Sam3Processor output: torch.Tensor or np.ndarray of shape (N, 1, H, W)
          or (N, H, W) — one entry per detected object.
        - A list / tuple of (H, W) arrays or tensors (original format).
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
            out.append(np.squeeze(m).astype(np.float32))
        return out

    @torch.inference_mode()
    def segment_volume(
        self,
        start_frame_idx: int,
        masks=None,
        vol_shape: Optional[Tuple[int, int, int]] = None,
        max_frame_num_to_track: Optional[int] = None,
        min_presence_score: float = 0.5,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Propagate bidirectionally from start_frame_idx and return a labeled
        3D volume.  This is the primary segmentation entry point.

        Returns:
            vol_masks: (Z, H, W) uint16 numpy array.
                0 = background, 1..N = individual detected objects.
        """
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before segment_volume().")

        if vol_shape is None:
            if self._vol_shape is None:
                raise RuntimeError(
                    "vol_shape required when inference_state is passed explicitly."
                )
            vol_shape = self._vol_shape

        Z, H, W = vol_shape

        # Seed from external 2D masks (processor output or list format)
        mask_list = self._normalize_masks(masks)
        for obj_id, mask in enumerate(mask_list, start=1):
            self.add_new_mask(
                frame_idx=start_frame_idx,
                obj_id=obj_id,
                mask=mask,
                inference_state=state,
            )

        vol_masks = np.zeros((Z, H, W), dtype=np.uint16)
        frame_metrics = self.frame_metrics

        def _apply(frame_idx: int, obj_ids, video_res_masks, obj_scores) -> None:
            if video_res_masks is None or len(obj_ids) == 0:
                return
            if frame_idx not in frame_metrics:
                frame_metrics[frame_idx] = {}
            for i, obj_id in enumerate(obj_ids):
                presence_score = (
                    float(torch.sigmoid(obj_scores[i]).item())
                    if obj_scores is not None
                    else 1.0
                )
                logits = video_res_masks[i]
                if hasattr(logits, "cpu"):
                    logits = logits.cpu().numpy()
                logits = np.squeeze(logits).astype(np.float32)  # (H, W)
                frame_metrics[frame_idx][obj_id] = {
                    "logits": logits,
                    "presence_score": presence_score,
                }

                if presence_score < min_presence_score:
                    continue
                m = logits > 0.0  # (H, W) bool
                if m.shape != (H, W):
                    import skimage.transform
                    m = skimage.transform.resize(
                        m, (H, W), order=0, anti_aliasing=False
                    )
                vol_masks[frame_idx] = np.where(
                    m, int(obj_id), vol_masks[frame_idx]
                )

        # Forward pass
        for frame_idx, obj_ids, _lrm, video_res_masks, obj_scores in (
            self.propagate_in_video(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=False,
                inference_state=state,
            )
        ):
            _apply(frame_idx, obj_ids, video_res_masks, obj_scores)

        # Backward pass (only fills slices not already set by forward)
        for frame_idx, obj_ids, _lrm, video_res_masks, obj_scores in (
            self.propagate_in_video(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=True,
                inference_state=state,
            )
        ):
            if not vol_masks[frame_idx].any():
                _apply(frame_idx, obj_ids, video_res_masks, obj_scores)

        return vol_masks

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(
        self, inference_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Clear all prompts and tracking state, keeping the loaded volume."""
        state = inference_state or self.inference_state
        if state is not None:
            self.predictor.clear_all_points_in_video(state)

    def remove_object(
        self, obj_id: int, inference_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Remove a single tracked object from the inference state."""
        state = inference_state or self.inference_state
        if state is not None:
            self.predictor.remove_object(inference_state=state, obj_id=obj_id)
