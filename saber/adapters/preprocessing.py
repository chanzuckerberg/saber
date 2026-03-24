from typing import Optional, Tuple
import skimage.transform
from tqdm import tqdm
import numpy as np
import torch

class TomogramPreprocessor:
    """
    MIT Licensed preprocessing utilities for tomogram data.
    This class handles tomogram-specific preprocessing without inheriting from SAM2.
    """

    def __init__(self, light_modality: bool = False):
        self.light_modality = light_modality

    def load_img_as_tensor(self, img: np.ndarray, image_size: int) -> Tuple[torch.Tensor, int, int]:
        """
        Convert a single 2D image to tensor format.
        Normalizing to [-1,1] to start with.
        """
        img = skimage.transform.resize(img, (image_size, image_size), anti_aliasing=True)
        img = np.repeat(img[None, ...], axis=0, repeats=3)
        img = torch.as_tensor(img, dtype=torch.float32)
        _, video_width, video_height = img.shape
        return img, video_height, video_width

    def load_grayscale_image_array(
        self,
        img_array: np.ndarray,
        image_size: int,
        offload_video_to_cpu: bool = False,
        img_mean: Optional[np.ndarray] = None,
        img_std: Optional[np.ndarray] = None,
        compute_device: torch.device = torch.device("cuda")
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Load image frames from a 3D numpy array (tomogram).
        """
        if img_mean is not None:
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        if img_std is not None:
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        images = torch.zeros(img_array.shape[0], 3, image_size, image_size, dtype=torch.float32)

        for n in tqdm(range(img_array.shape[0]), desc="Loading tomogram slices"):
            images[n], video_height, video_width = self.load_img_as_tensor(
                img_array[n], image_size
            )

        if not offload_video_to_cpu:
            images = images.to(compute_device)
            if img_mean is not None:
                img_mean = img_mean.to(compute_device)
            if img_std is not None:
                img_std = img_std.to(compute_device)

        if img_mean is None and img_std is None:
            images = 2 * images - 1
        else:
            if img_mean is not None:
                images -= img_mean
            if img_std is not None:
                images /= img_std

        if self.light_modality:
            images = (images - images.min()) / (images.max() - images.min())
            images *= 255

        return images, video_height, video_width

    def normalize_tomogram(self, tomogram: np.ndarray) -> np.ndarray:
        """Normalize tomogram to [0, 1] range, then to [-1, 1]."""
        tomogram = (tomogram - tomogram.min()) / (tomogram.max() - tomogram.min())
        tomogram = tomogram * 2 - 1
        return tomogram
