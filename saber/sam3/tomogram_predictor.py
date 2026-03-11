from saber.sam2.tomogram_predictor import TomogramPreprocessor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


class TomogramSAM3Adapter:
    """
    Adapter for running SAM3 segmentation on tomogram data.

    Mirrors the TomogramSAM2Adapter interface so the two can be swapped.
    Uses SAM3's SAM2-compatible tracker API (Sam3TrackerPredictor) directly,
    which is the same interface shown in sam3_for_sam2_video_task_example.ipynb.

    Typical workflow
    ----------------
    >>> adapter = TomogramSAM3Adapter()
    >>> adapter.set_volume(vol)                       # (Z, H, W) numpy array
    >>> adapter.add_new_mask(zSlice, obj_id=1, mask)  # binary (H, W) numpy array
    >>> vol_masks = adapter.segment_volume(zSlice)    # (Z, H, W) uint16

    Or with box/point prompts:
    >>> adapter.set_volume(vol)
    >>> adapter.add_box_prompt(zSlice, obj_id=1, box_xyxy_norm=[x0,y0,x1,y1])
    >>> vol_masks = adapter.segment_volume(zSlice)
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        load_from_HF: bool = True,
        light_modality: bool = False,
    ):
        """
        Args:
            device: PyTorch device string (e.g. "cuda", "cuda:1").
            checkpoint_path: Path to a local SAM3 checkpoint (.pt).
                If None and load_from_HF is True, downloaded from
                HuggingFace (facebook/sam3) automatically.
            load_from_HF: Download weights from HuggingFace when no local
                checkpoint is provided.  Requires `huggingface-cli login`
                or the HF_TOKEN environment variable.
            light_modality: Set True for fluorescence / light microscopy data.
        """
        from sam3.model_builder import build_sam3_video_model
        from saber.pretrained_weights import get_sam3_bpe_path

        sam3_model = (
            build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_HF,
                bpe_path=get_sam3_bpe_path(),
            )
            .to(device)
            .eval()
        )

        # Use the SAM2-compatible tracker directly (as shown in the notebook)
        self.predictor = sam3_model.tracker
        self.predictor.backbone = sam3_model.detector.backbone

        self.device = torch.device(device)
        self.preprocessor = TomogramPreprocessor(light_modality)

        # Internal state — set by set_volume()
        self.inference_state: Optional[Dict[str, Any]] = None
        self._vol_shape: Optional[Tuple[int, int, int]] = None

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

        Args:
            tomogram: (Z, H, W) numpy array.  Any numeric dtype accepted.
            offload_video_to_cpu: Keep frame tensors in CPU memory to save GPU.
        """
        self._vol_shape = tomogram.shape
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

        Pipeline:
          1. Min-max normalize to [0, 1]
          2. Resize each slice to image_size × image_size
          3. Repeat grayscale to 3 channels
          4. Apply 2x - 1  →  [-1, 1]   (same as SAM3's load_video_frames)
        """
        t_min = float(tomogram.min())
        t_max = float(tomogram.max())
        if t_max > t_min:
            norm01 = (tomogram.astype(np.float32) - t_min) / (t_max - t_min)
        else:
            norm01 = np.zeros_like(tomogram, dtype=np.float32)

        # load_grayscale_image_array resizes + applies 2x-1 by default
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

        Uses Sam3TrackerPredictor.init_state() with injected pre-built
        image tensors instead of reading from a video file on disk.
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
        """
        Seed segmentation on a Z-slice using a binary mask.

        This is the preferred prompt type when masks come from an external
        2D segmenter (e.g. Sam3Processor or the existing cryoTomoSegmenter).

        Args:
            frame_idx: Z-slice index to seed on.
            obj_id: Unique integer ID for this object (must be unique per
                object across the volume).
            mask: (H, W) boolean or float numpy array.  Any non-zero value
                is treated as foreground.
            inference_state: Override the internal state set by set_volume().

        Returns:
            (frame_idx, obj_ids, low_res_masks, video_res_masks)
        """
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
    def add_box_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        box_xyxy_norm: List[float],
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any, Any, Any]:
        """
        Seed segmentation on a Z-slice using a bounding box.

        Args:
            frame_idx: Z-slice index to seed on.
            obj_id: Unique integer ID for this object.
            box_xyxy_norm: [x_min, y_min, x_max, y_max] normalized to [0, 1]
                relative to the slice dimensions.
            inference_state: Override the internal state set by set_volume().

        Returns:
            (frame_idx, obj_ids, low_res_masks, video_res_masks)
        """
        state = inference_state or self.inference_state
        if state is None:
            raise RuntimeError("Call set_volume() before add_box_prompt().")
        rel_box = np.array([box_xyxy_norm], dtype=np.float32)
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=rel_box,
            # rel_coordinates=True (default): tracker converts [0,1] → image_size coords
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
        """
        Seed segmentation on a Z-slice using point clicks.

        Args:
            frame_idx: Z-slice index to seed on.
            obj_id: Unique integer ID for this object.
            points_norm: (N, 2) float array of [x, y] coordinates normalized
                to [0, 1] relative to the slice dimensions.
            labels: (N,) int32 array; 1 = positive, 0 = negative click.
            inference_state: Override the internal state set by set_volume().

        Returns:
            (frame_idx, obj_ids, low_res_masks, video_res_masks)
        """
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
            # (N, 1, H, W) or (N, H, W) — batch of masks from the processor
            return [np.squeeze(masks[i]).astype(np.float32) for i in range(masks.shape[0])]

        # Already an iterable of (H, W) arrays / tensors
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
        min_score: float = 0.0,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Propagate bidirectionally from start_frame_idx and return a labeled
        3D volume.  This is the primary segmentation entry point.

        Args:
            start_frame_idx: Seed Z-slice (must have been prompted first,
                OR pass initial masks via the ``masks`` argument).
            masks: Optional seed masks for start_frame_idx.  Accepts:

                * **Sam3Processor format** — a tensor/array of shape
                  ``(N, 1, H, W)`` or ``(N, H, W)`` as returned by
                  ``processor.set_text_prompt()``.  Each of the N detections
                  becomes a separate tracked object (obj_id = 1 … N)::

                      output = processor.set_text_prompt(state, "ribosome")
                      vol = adapter.segment_volume(z, masks=output["masks"])

                * **List format** — a list of ``(H, W)`` binary numpy arrays
                  or tensors, one per object.

                If ``None``, prompts must have been added beforehand via
                ``add_new_mask()`` / ``add_box_prompt()`` / ``add_point_prompt()``.

            vol_shape: (Z, H, W) of the output volume.  Inferred from
                set_volume() if not provided.
            max_frame_num_to_track: Maximum slices per direction.
                None = propagate to volume boundary.
            min_score: Minimum object score logit to include in the output.
                0.0 keeps all detections (sigmoid > 0.5 → positive objects).
            inference_state: Override the internal state set by set_volume().

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

        def _apply(frame_idx: int, obj_ids, video_res_masks, obj_scores) -> None:
            if video_res_masks is None or len(obj_ids) == 0:
                return
            for i, obj_id in enumerate(obj_ids):
                # obj_scores: (batch, 1) logits; sigmoid > 0.5 means object present
                if obj_scores is not None:
                    score = float(obj_scores[i].item())
                    if score < min_score:
                        continue
                m = video_res_masks[i]
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                m = np.squeeze(m) > 0.0  # (H, W) bool
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
