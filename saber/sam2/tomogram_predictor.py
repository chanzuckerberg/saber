from sam2.build_sam import build_sam2_video_predictor
from typing import Optional, Tuple, Any, Dict
from collections import OrderedDict
import skimage.transform
from tqdm import tqdm
import numpy as np
import torch

class TomogramPreprocessor:
    """
    MIT Licensed preprocessing utilities for tomogram data.
    This class handles tomogram-specific preprocessing without inheriting from SAM2.
    """
    
    def __init__(self):
        pass
    
    def load_img_as_tensor(self, img: np.ndarray, image_size: int) -> Tuple[torch.Tensor, int, int]:
        """
        Convert a single 2D image to tensor format.
        Normalizing to [-1,1] to start with.
        
        Args:
            img: 2D numpy array representing a single slice
            image_size: Target size for resizing
            
        Returns:
            Tuple of (tensor, video_height, video_width)
        """
        # Resize the image
        img = skimage.transform.resize(img, (image_size, image_size), anti_aliasing=True)
        
        # Convert grayscale to RGB by repeating channel
        img = np.repeat(img[None, ...], axis=0, repeats=3)
        
        # Convert to tensor
        img = torch.as_tensor(img, dtype=torch.float32)
        
        # Get original dimensions (they're the same after resize, but keeping for compatibility)
        _, video_width, video_height = img.shape
        
        return img, video_height, video_width
    
    def load_grayscale_image_array(
        self,
        img_array: np.ndarray,
        image_size: int,
        offload_video_to_cpu: bool = False,
        img_mean: Optional[np.ndarray] = None,
        img_std: Optional[np.ndarray] = None,
        compute_device: torch.device = torch.device("cuda"),
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Load image frames from a 3D numpy array (tomogram).
        
        Args:
            img_array: 3D numpy array of shape (num_slices, height, width)
            image_size: Target size for each image
            offload_video_to_cpu: Whether to keep tensors on CPU
            img_mean: Optional normalization mean
            img_std: Optional normalization std
            compute_device: Target device for tensors
            
        Returns:
            Tuple of (images_tensor, video_height, video_width)
        """
        if img_mean is not None:
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        if img_std is not None:
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        # Pre-allocate tensor for all images
        images = torch.zeros(img_array.shape[0], 3, image_size, image_size, dtype=torch.float32)
        
        # Process each slice
        for n in tqdm(range(img_array.shape[0]), desc="Loading tomogram slices"):
            images[n], video_height, video_width = self.load_img_as_tensor(
                img_array[n], image_size
            )
        
        # Move to appropriate device
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            if img_mean is not None:
                img_mean = img_mean.to(compute_device)
            if img_std is not None:
                img_std = img_std.to(compute_device)

        # Apply normalization
        if img_mean is None and img_std is None:
            # Default normalization to [-1, 1]
            images = 2 * images - 1
        else:
            if img_mean is not None:
                images -= img_mean
            if img_std is not None:
                images /= img_std

        return images, video_height, video_width
    
    def normalize_tomogram(self, tomogram: np.ndarray) -> np.ndarray:
        """
        Normalize tomogram to [0, 1] range, then to [-1, 1].
        
        Args:
            tomogram: 3D numpy array
            
        Returns:
            Normalized tomogram
        """
        # Normalize to [0, 1]
        tomogram = (tomogram - tomogram.min()) / (tomogram.max() - tomogram.min())
        # Convert to [-1, 1]
        tomogram = tomogram * 2 - 1
        return tomogram


class TomogramSAM2Adapter:
    """
    MIT Licensed adapter that provides a clean interface between tomogram data and SAM2.
    This uses composition instead of inheritance to avoid license mixing.
    """
    
    def __init__(self, cfg, checkpoint, device, num_maskmem: int = 2):
        """
        Initialize with a SAM2 predictor instance.
        
        Args:
            sam2_predictor: An instance of SAM2VideoPredictor
        """

        # Build SAM2 Video Predictor
        self.predictor = build_sam2_video_predictor(
            cfg, checkpoint, device=device, vos_optimized=False,
        )

        # Check to make sure num_maskmem is less than 7
        if num_maskmem > 7:
            raise ValueError("num_maskmem must be less than 7")

        # Adjust the Number of Mask Memory Frames
        maskmem = self.predictor.maskmem_tpos_enc[:num_maskmem]
        self.predictor.maskmem_tpos_enc = torch.nn.Parameter(maskmem)
        # update the num_maskmem attribute if it exists
        if hasattr(self.predictor, 'num_maskmem'):
            self.predictor.num_maskmem = num_maskmem  

        # Initialize Preprocessor
        self.preprocessor = TomogramPreprocessor()
    
    @torch.inference_mode()
    def create_inference_state_from_tomogram(
        self,
        tomogram: np.ndarray,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Create inference state from tomogram data.
        
        Args:
            tomogram: 3D numpy array of shape (num_slices, height, width)
            offload_video_to_cpu: Whether to store frames on CPU
            offload_state_to_cpu: Whether to store inference state on CPU
            
        Returns:
            Inference state dictionary compatible with SAM2
        """
        # Normalize the tomogram
        normalized_tomogram = self.preprocessor.normalize_tomogram(tomogram)
        
        # Convert to tensor format
        images, video_height, video_width = self.preprocessor.load_grayscale_image_array(
            normalized_tomogram,
            image_size=self.predictor.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=self.predictor.device,
        )
        
        # Create inference state structure
        inference_state = self._create_empty_inference_state(
            images=images,
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        
        # Warm up the visual backbone
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
        """
        Create an empty inference state structure compatible with SAM2.
        This replicates the structure without inheriting the code.
        """
        compute_device = self.predictor.device
        
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device}

        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}

        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}

        # Warm up the visual backbone and cache the image feature on frame 0
        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        
        return inference_state
    
    # Delegate all other methods to the underlying predictor
    def add_new_points_or_box(self, *args, **kwargs):
        """Add new points or box - delegates to SAM2 predictor."""
        return self.predictor.add_new_points_or_box(*args, **kwargs)
    
    def add_new_mask(self, *args, **kwargs):
        """Add new mask - delegates to SAM2 predictor."""
        return self.predictor.add_new_mask(*args, **kwargs)
    
    @torch.inference_mode()
    def propagate_in_video(self, *args, **kwargs):
        """Propagate tracking in video - delegates to SAM2 predictor."""
        return self.predictor.propagate_in_video(*args, **kwargs)
    
    def clear_all_prompts_in_frame(self, *args, **kwargs):
        """Clear prompts - delegates to SAM2 predictor."""
        return self.predictor.clear_all_prompts_in_frame(*args, **kwargs)
    
    def reset_state(self, *args, **kwargs):
        """Reset state - delegates to SAM2 predictor."""
        return self.predictor.reset_state(*args, **kwargs)
    
    def remove_object(self, *args, **kwargs):
        """Remove object - delegates to SAM2 predictor."""
        return self.predictor.remove_object(*args, **kwargs)