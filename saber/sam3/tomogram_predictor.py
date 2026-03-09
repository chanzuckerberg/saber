from saber.sam2.tomogram_predictor import TomogramPreprocessor
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch


class TomogramSAM3Adapter:
    """
    Adapter for running SAM3 segmentation on tomogram data.

    Bypasses SAM3's video file reader and directly injects preprocessed
    numpy arrays as float16 tensors — the same approach used in
    TomogramSAM2Adapter.  Z-slices are treated as video frames, allowing
    SAM3's tracker to propagate segmentations through the volume.

    Typical workflow
    ----------------
    1. Build adapter (downloads weights from HuggingFace on first use)
    2. Create inference state from a (Z, H, W) tomogram array
    3. Add one or more text / box prompts on a seed Z-slice
    4. Propagate bidirectionally through Z
    5. Collect per-slice mask outputs

    Example
    -------
    >>> adapter = TomogramSAM3Adapter(device="cuda")
    >>> state = adapter.create_inference_state_from_tomogram(tomogram)  # (Z,H,W)
    >>> adapter.add_text_prompt(state, frame_idx=10, text="organelle")
    >>> results = adapter.propagate_bidirectional(state, start_frame_idx=10)
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
                If None and load_from_HF is True the checkpoint is
                downloaded from HuggingFace (facebook/sam3) automatically.
            load_from_HF: Download weights from HuggingFace when no local
                checkpoint is provided.  Requires `huggingface-cli login`
                or the HF_TOKEN environment variable.
            light_modality: Set True for fluorescence / light microscopy
                data (keeps pixel values in [0, 255] after normalization
                instead of [-1, 1]).
        """
        from sam3.model_builder import build_sam3_video_model
        from saber.pretrained_weights import get_sam3_bpe_path

        self.model = (
            build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_HF,
                bpe_path=get_sam3_bpe_path(),
            )
            .to(device)
            .eval()
        )
        self.device = torch.device(device)
        self.preprocessor = TomogramPreprocessor(light_modality)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_tomogram(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Normalize a (Z, H, W) tomogram and convert it to the float16
        tensor format expected by SAM3's BatchedDatapoint.

        SAM3 normalises images with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        which maps [0, 1] → [-1, 1].  TomogramPreprocessor.normalize_tomogram
        already maps data to [0, 1] then to [-1, 1], so the two are equivalent.

        Returns:
            images : (Z, 3, image_size, image_size) float16 tensor
            orig_height, orig_width : original spatial dimensions
        """
        normalized = self.preprocessor.normalize_tomogram(tomogram)

        images, orig_height, orig_width = self.preprocessor.load_grayscale_image_array(
            normalized,
            image_size=self.model.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=self.device,
        )

        # SAM3 stores frames as float16 internally (matching load_resource_as_video_frames)
        images = images.to(torch.float16)
        if offload_video_to_cpu:
            images = images.cpu()

        return images, orig_height, orig_width

    # ------------------------------------------------------------------
    # Inference state
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def create_inference_state_from_tomogram(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a SAM3 inference state directly from a tomogram array,
        bypassing the video file reader.

        Args:
            tomogram: 3-D numpy array of shape (Z, H, W).  Any numeric
                dtype is accepted; values are normalised internally.
            offload_video_to_cpu: Keep frame tensors in CPU memory to
                reduce GPU memory usage (at the cost of slightly slower
                inference).

        Returns:
            inference_state dict ready for add_text_prompt / propagate_in_video.
        """
        images, orig_height, orig_width = self._preprocess_tomogram(
            tomogram, offload_video_to_cpu
        )

        # Mirror Sam3VideoInference.init_state() but inject our tensor
        # instead of reading from a video file.
        inference_state: Dict[str, Any] = {}
        inference_state["image_size"] = self.model.image_size
        inference_state["num_frames"] = images.shape[0]
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        inference_state["constants"] = {}

        # Populate the BatchedDatapoint and all per-frame placeholder lists.
        # _construct_initial_input_batch expects images as (N, 3, H, W).
        self.model._construct_initial_input_batch(inference_state, images)

        # Remaining keys initialised by init_state() after _construct_initial_input_batch
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []
        inference_state["is_image_only"] = False  # always 3-D video mode for tomograms

        return inference_state

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def add_text_prompt(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        text: str,
        obj_id: Optional[int] = None,
    ) -> Tuple[int, Any]:
        """
        Seed segmentation on a Z-slice using a text prompt.

        Args:
            inference_state: Returned by create_inference_state_from_tomogram.
            frame_idx: Index of the Z-slice to seed on.
            text: Open-vocabulary text description of the structure to find,
                e.g. "organelle", "vesicle", "membrane", "ribosome".
            obj_id: Optional integer object identifier.  If None, one is
                assigned automatically.

        Returns:
            (frame_idx, outputs) — outputs contains the detected masks on
            this slice in SAM3's internal format.
        """
        return self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            obj_id=obj_id,
        )

    @torch.inference_mode()
    def add_box_prompt(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        box_xywh: list,
        obj_id: Optional[int] = None,
    ) -> Tuple[int, Any]:
        """
        Seed segmentation on a Z-slice using a bounding box.

        Args:
            inference_state: Returned by create_inference_state_from_tomogram.
            frame_idx: Index of the Z-slice to seed on.
            box_xywh: Bounding box as [x, y, width, height] in pixel
                coordinates of the slice.
            obj_id: Optional integer object identifier.

        Returns:
            (frame_idx, outputs) — outputs contains the mask on this slice.
        """
        return self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            boxes_xywh=[box_xywh],
            box_labels=[1],
            obj_id=obj_id,
        )

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state: Dict[str, Any],
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ) -> Iterator[Tuple[int, Any]]:
        """
        Propagate segmentation through Z-slices in one direction.

        Args:
            inference_state: Returned by create_inference_state_from_tomogram.
            start_frame_idx: Z-slice to start from (defaults to the first
                prompted frame).
            max_frame_num_to_track: Maximum number of slices to propagate.
                None means propagate to the end of the volume.
            reverse: If True, propagate toward lower Z indices.

        Yields:
            (frame_idx, outputs) for each processed Z-slice.
        """
        yield from self.model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

    @torch.inference_mode()
    def propagate_bidirectional(
        self,
        inference_state: Dict[str, Any],
        start_frame_idx: int,
        max_frame_num_to_track: Optional[int] = None,
    ) -> Dict[int, Any]:
        """
        Propagate both forward and backward from start_frame_idx.

        Forward results take priority; backward results fill in any
        slices not reached by forward propagation.

        Args:
            inference_state: Returned by create_inference_state_from_tomogram.
            start_frame_idx: Seed Z-slice (must have been prompted first).
            max_frame_num_to_track: Maximum slices to propagate in each
                direction.  None means propagate to the volume boundary.

        Returns:
            Dict mapping frame_idx → outputs for every propagated slice.
        """
        results: Dict[int, Any] = {}

        for frame_idx, outputs in self.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
        ):
            results[frame_idx] = outputs

        for frame_idx, outputs in self.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=True,
        ):
            results.setdefault(frame_idx, outputs)

        return results

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, inference_state: Dict[str, Any]) -> None:
        """Reset inference state to post-load condition, clearing all prompts."""
        self.model.reset_state(inference_state)

    def remove_object(self, inference_state: Dict[str, Any], obj_id: int) -> None:
        """Remove a tracked object from the inference state."""
        self.model.remove_object(inference_state=inference_state, obj_id=obj_id)
