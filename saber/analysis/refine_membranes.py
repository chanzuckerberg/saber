"""
Organelle-Membrane Segmentation Filtering Pipeline

This module provides a GPU-optimized pipeline for filtering and refining organelle and membrane 
segmentation masks. The core approach uses a "combined mask" strategy where membrane pixels are 
subtracted from organelle pixels to create an interior mask that is then refined using morphological 
operations.

Key Features:
- GPU-accelerated morphological operations using PyTorch
- Batch processing for efficient handling of multiple organelles
- Membrane-guided organelle constraint to prevent over-segmentation
- Robust edge trimming and small object removal
- Automatic fallback to CPU operations when GPU is unavailable

Pipeline Overview:
1. Preprocess membrane masks (trim edges, remove small objects)
2. Filter organelles by membrane presence (remove organelles without nearby membranes)
3. For each organelle:
   - Extract ROI (region of interest) with padding
   - Enhance membrane by slight dilation within organelle vicinity
   - Create combined mask by subtracting membrane from organelle
   - Apply morphological opening to clean up the combined mask
   - Use cleaned combined mask to constrain both organelle and membrane
   - Keep only largest connected components

The combined mask approach ensures that:
- Organelles are constrained to stay within membrane boundaries
- Membranes are refined to be associated with their corresponding organelles
- Both segmentations are topologically consistent

Usage:
    config = FilteringConfig(ball_size=5, min_membrane_area=10000)
    filter_obj = OrganelleMembraneFilter(config)
    results = filter_obj.filter_organelle_membrane_segmentation(organelle_seg, membrane_seg)
    
Author: [Your Name]
Date: [Current Date]
"""

from typing import Union, Tuple, Dict, Optional, List
from dataclasses import dataclass
import scipy.ndimage as ndi
import torch.nn.functional as F
from tqdm import tqdm
import logging, torch
import numpy as np

logger = logging.getLogger(__name__)

TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class FilteringConfig:
    """Configuration for organelle-membrane filtering pipeline."""
    ball_size: int = 3  # Reduced from 5 to be gentler
    min_membrane_area: int = 10000
    edge_trim_z: int = 5
    edge_trim_xy: int = 3
    min_roi_relative_size: float = 0.15
    batch_size: int = 8  # For GPU batch processing
    keep_surface_membranes: bool = False


class OrganelleMembraneFilter:
    """
    Filters organelle and membrane segmentations with GPU-optimized main processing loop.
    """
    
    def __init__(self, config: FilteringConfig = None, gpu_id: int = None):
        self.config = config or FilteringConfig()
        self._ball_kernel_cache = {}
        
        # Device selection logic
        if gpu_id is not None:
            # Specific GPU ID provided
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                self.device = torch.device(f'cuda:{gpu_id}')
                logger.info(f"Using GPU {gpu_id}")
            else:
                logger.warning(f"GPU {gpu_id} not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            # No specific GPU ID provided, auto-detect
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  # Use first available GPU
                logger.info(f"Auto-detected GPU, using cuda:0")
            else:
                self.device = torch.device('cpu')
                logger.info("No GPU available, using CPU")
    
    def _get_ball_kernel(self, radius: int) -> torch.Tensor:
        """Get cached ball kernel or create new one."""
        cache_key = (radius, self.device)
        if cache_key not in self._ball_kernel_cache:
            self._ball_kernel_cache[cache_key] = self._create_ball_kernel(radius)
        return self._ball_kernel_cache[cache_key]
    
    def _create_ball_kernel(self, radius: int) -> torch.Tensor:
        """Create 3D ball-shaped structuring element."""
        size = 2 * radius + 1
        center = radius
        
        # Create coordinate grids
        z, y, x = torch.meshgrid(
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device), 
            torch.arange(size, device=self.device),
            indexing='ij'
        )
        
        # Calculate distance from center
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
        kernel = (dist_sq <= radius**2).float()
        
        return kernel
    
    def _trim_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """Trim edges to remove boundary artifacts."""
        # Trim Z edges
        trimmed = torch.zeros_like(mask)
        z_trim = self.config.edge_trim_z
        if z_trim < mask.shape[0] // 2:
            trimmed[z_trim:-z_trim] = mask[z_trim:-z_trim]
        
        # Trim XY edges
        mask = trimmed
        trimmed = torch.zeros_like(mask)
        xy_trim = self.config.edge_trim_xy
        if xy_trim < mask.shape[1] // 2 and xy_trim < mask.shape[2] // 2:
            trimmed[:, xy_trim:-xy_trim, xy_trim:-xy_trim] = mask[:, xy_trim:-xy_trim, xy_trim:-xy_trim]
        
        return trimmed
    
    def _remove_small_objects(self, mask: torch.Tensor, min_size: int) -> torch.Tensor:
        """Remove connected components smaller than min_size using scipy."""
        if mask.sum() == 0:
            return mask
        
        # Convert to numpy for scipy processing
        device = mask.device
        binary_mask = (mask > 0).cpu().numpy()
        
        # Get 3D connected components
        labels, num_labels = ndi.label(binary_mask)
        
        # Find components to keep
        unique_labels, counts = np.unique(labels, return_counts=True)
        keep_labels = unique_labels[(unique_labels != 0) & (counts >= min_size)]
        
        if len(keep_labels) == 0:
            return torch.zeros_like(mask)
        
        # Create mask for keeping only large objects
        keep_mask = np.isin(labels, keep_labels)
        result = mask * torch.from_numpy(keep_mask).to(device)
        
        return result
    
    def _keep_surface_membranes_only(self, membrane_mask: torch.Tensor, organelle_mask: torch.Tensor) -> torch.Tensor:
        """Keep only membrane components that are on the organelle surface, remove internal ones."""
        if membrane_mask.sum() == 0:
            return membrane_mask
        
        device = membrane_mask.device
        membrane_np = (membrane_mask > 0).cpu().numpy()
        organelle_np = (organelle_mask > 0).cpu().numpy()
        
        # Get organelle boundary (surface) using erosion
        # The boundary is: original - eroded
        eroded_org = ndi.binary_erosion(organelle_np, structure=np.ones((3, 3, 3)))
        organelle_boundary = organelle_np & ~eroded_org
        
        # Label individual membrane components
        membrane_labels, num_labels = ndi.label(membrane_np)
        
        if num_labels == 0:
            return membrane_mask
        
        # Check each membrane component
        surface_membrane_mask = np.zeros_like(membrane_np)
        
        for label_id in range(1, num_labels + 1):
            component_mask = membrane_labels == label_id
            
            # Check if this component overlaps with organelle boundary
            overlap_with_boundary = np.sum(component_mask & organelle_boundary)
            component_size = np.sum(component_mask)
            
            # Keep component if significant portion is on boundary
            boundary_ratio = overlap_with_boundary / component_size if component_size > 0 else 0
            
            if boundary_ratio > 0.1:  # At least 30% of membrane component should be on boundary
                surface_membrane_mask |= component_mask
        
        # Convert back to torch with original labels preserved
        surface_membrane_torch = torch.from_numpy(surface_membrane_mask.astype(np.uint8)).to(device)
        return membrane_mask * surface_membrane_torch

    def _remove_small_membrane_components(self, mask: torch.Tensor, min_size: int = 100) -> torch.Tensor:
        """Remove small membrane components but keep multiple large ones (e.g., both sides of organelle)."""
        if mask.sum() == 0:
            return mask
        
        device = mask.device
        binary_mask = (mask > 0).cpu().numpy()
        
        labels, num_labels = ndi.label(binary_mask)
        
        if num_labels == 0:
            return mask
        
        # Find components to keep (all components above min_size)
        unique_labels, counts = np.unique(labels, return_counts=True)
        keep_labels = unique_labels[(unique_labels != 0) & (counts >= min_size)]
        
        if len(keep_labels) == 0:
            return torch.zeros_like(mask)
        
        # Keep all large components, not just the largest
        keep_mask = np.isin(labels, keep_labels)
        
        return mask * torch.from_numpy(keep_mask).to(device)

    def _get_largest_component(self, mask: torch.Tensor) -> torch.Tensor:
        """Keep only the largest connected component using scipy."""
        if mask.sum() == 0:
            return mask
        
        device = mask.device
        binary_mask = (mask > 0).cpu().numpy()
        
        labels, num_labels = ndi.label(binary_mask)
        
        if num_labels == 0:
            return mask
        
        # Find largest component
        unique_labels, counts = np.unique(labels, return_counts=True)
        non_zero_mask = unique_labels != 0
        
        if not non_zero_mask.any():
            return mask
        
        largest_label = unique_labels[non_zero_mask][counts[non_zero_mask].argmax()]
        largest_mask = labels == largest_label
        
        return mask * torch.from_numpy(largest_mask).to(device)
    
    def _get_organelle_roi(self, organelle_mask: torch.Tensor, pad: int) -> Optional[Tuple[int, ...]]:
        """Get bounding box ROI for organelle with padding."""
        nonzero_indices = torch.nonzero(organelle_mask, as_tuple=False)
        if len(nonzero_indices) == 0:
            return None
        
        mins = nonzero_indices.min(dim=0)[0]
        maxs = nonzero_indices.max(dim=0)[0] + 1  # Make inclusive
        
        # Check minimum size requirement
        sizes = maxs - mins
        shape = torch.tensor(organelle_mask.shape, device=organelle_mask.device)
        min_sizes = self.config.min_roi_relative_size * shape
        
        if (sizes < min_sizes).any():
            return None
        
        # Apply padding and clamp to bounds
        mins = torch.clamp(mins - pad, 0)
        maxs = torch.clamp(maxs + pad, max=shape)
        
        return tuple(mins.tolist() + maxs.tolist())

    def _torch_erosion_3d(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """3D erosion using conv3d."""
        if image.sum() == 0:
            return image
        
        # Pad image
        pad_size = kernel.shape[0] // 2
        padded = F.pad(image, [pad_size]*6, mode='constant', value=0)
        
        # Prepare kernel for convolution
        kernel_conv = kernel.flip([0, 1, 2]).unsqueeze(0).unsqueeze(0)
        
        # Erosion: convolve and check if all kernel pixels are covered
        conv_result = F.conv3d(padded.unsqueeze(0).unsqueeze(0), kernel_conv, padding=0)
        kernel_sum = kernel.sum()
        
        return (conv_result.squeeze() >= kernel_sum - 1e-6).float()
    
    def _torch_dilation_3d(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """3D dilation using conv3d."""
        if image.sum() == 0:
            return image
        
        # Pad image
        pad_size = kernel.shape[0] // 2
        padded = F.pad(image, [pad_size]*6, mode='constant', value=0)
        
        # Prepare kernel for convolution
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0)
        
        # Dilation: convolve and check if any kernel pixel overlaps
        conv_result = F.conv3d(padded.unsqueeze(0).unsqueeze(0), kernel_conv, padding=0)
        
        return (conv_result.squeeze() > 1e-6).float()
    
    def _morphological_opening(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Morphological opening - use GPU version if available, fallback to scipy."""
        if self.device.type == 'cuda':
            return self._morphological_opening_gpu(image, kernel)
        else:
            # Fallback to scipy for CPU
            from scipy import ndimage
            image_np = image.cpu().numpy()
            kernel_np = kernel.cpu().numpy()
            opened_np = ndimage.binary_opening(image_np > 0, structure=kernel_np > 0).astype(image_np.dtype)
            return torch.from_numpy(opened_np).to(image.device)

    def _morphological_opening_gpu(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """GPU-optimized morphological opening using conv3d."""
        if image.sum() == 0:
            return image
        
        # Convert to binary
        binary_img = (image > 0).float()
        
        # Perform erosion then dilation using conv3d
        eroded = self._torch_erosion_3d(binary_img, kernel)
        opened = self._torch_dilation_3d(eroded, kernel)
        
        return opened

    def _process_organelle_batch(
        self, 
        organelle_masks: torch.Tensor,
        membrane_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process a batch of organelles using combined mask approach."""
        batch_size = organelle_masks.shape[0]
        ball_size = self.config.ball_size

        keep_surface_membranes = self.config.keep_surface_membranes

        results = []
        for i in range(batch_size):
            org_mask = organelle_masks[i]
            label = labels[i]
            
            # Get ROI
            roi = self._get_organelle_roi(org_mask, pad=ball_size // 2)
            if roi is None:
                results.append(None)
                continue
            
            minz, miny, minx, maxz, maxy, maxx = roi
            
            # Extract ROI regions
            org_roi = org_mask[minz:maxz, miny:maxy, minx:maxx]
            mem_roi = membrane_mask[minz:maxz, miny:maxy, minx:maxx]

            # Adaptive parameters based on organelle shape
            roi_shape = torch.tensor([maxz-minz, maxy-miny, maxx-minx])
            aspect_ratio = roi_shape.max().float() / roi_shape.min().float()
            
            # Use gentler parameters for elongated structures
            if aspect_ratio > 3.0:  # Elongated organelle
                dilate_size = 1
                morph_ball_size = max(1, ball_size // 2)
            else:  # More circular organelle
                dilate_size = 2
                morph_ball_size = ball_size

            # Enhance membrane by dilating slightly
            dilate_kernel = self._get_ball_kernel(dilate_size)
            dilated_membrane = self._torch_dilation_3d(mem_roi.float(), dilate_kernel)
            
            # Constrain to organelle vicinity
            organelle_expanded = self._torch_dilation_3d(org_roi.float(), dilate_kernel)
            enhanced_membrane = dilated_membrane * organelle_expanded
            
            if enhanced_membrane.sum() == 0:
                results.append(None)
                continue
            
            # Keep only largest connected component (but preserve multiple membrane components)
            cleaned_membrane = self._get_largest_component(enhanced_membrane)
            
            # BETTER: Keep multiple membrane components instead of just largest
            # This preserves both sides of elongated organelles
            cleaned_membrane = self._remove_small_membrane_components(enhanced_membrane, min_size=100)
            
            # Keep only surface membranes, remove internal ones
            if keep_surface_membranes:
                cleaned_membrane = self._keep_surface_membranes_only(cleaned_membrane, org_roi)
            
            if cleaned_membrane.sum() == 0:
                results.append(None)
                continue

            # Create combined mask by subtracting membrane from organelle
            mem_roi_int = cleaned_membrane.int()
            org_roi_int = org_roi.int()

            comb_mask_roi = org_roi_int - mem_roi_int
            comb_mask_roi[comb_mask_roi != 0] = torch.maximum(comb_mask_roi, torch.ones_like(comb_mask_roi))[comb_mask_roi != 0]

            # Apply morphological opening with adaptive ball size
            ball_kernel = self._get_ball_kernel(morph_ball_size)
            comb_mask_roi_out = self._morphological_opening(comb_mask_roi.float(), ball_kernel)

            if comb_mask_roi_out.sum() == 0:
                # Try without opening as fallback
                comb_mask_roi_out = comb_mask_roi.float()
                
                if comb_mask_roi_out.sum() == 0:
                    results.append(None)
                    continue

            # Keep only largest connected component
            comb_mask_roi_out = self._get_largest_component(comb_mask_roi_out)

            # Use combined mask to constrain organelle
            org_roi_out = org_roi * (comb_mask_roi_out > 0)
            organelle_cleaned_roi = self._get_largest_component(org_roi_out)

            # Use combined mask to constrain membrane (preserve multiple components)
            mem_roi_out = cleaned_membrane * (comb_mask_roi_out > 0)
            membrane_cleaned_roi = self._remove_small_membrane_components(mem_roi_out, min_size=50)  # Keep multiple components

            # Create full-size masks
            membrane_instance_full = torch.zeros_like(org_mask)
            if membrane_cleaned_roi.sum() > 0:
                membrane_instance_full[minz:maxz, miny:maxy, minx:maxx] = membrane_cleaned_roi
                membrane_instance_full[membrane_instance_full > 0] = label - 1

            organelle_full = torch.zeros_like(org_mask)
            organelle_full[minz:maxz, miny:maxy, minx:maxx] = organelle_cleaned_roi

            results.append((membrane_instance_full, organelle_full))
        return results

    def run(
        self,
        organelle_seg: TensorLike,
        membrane_seg: TensorLike, 
        batch_processing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Main filtering pipeline with GPU-optimized processing loop."""
        
        # Suppress the Logger 
        if batch_processing:
            # logger = logging.getLogger()
            logger.disabled = True

        logger.info("Starting organelle-membrane filtering pipeline")
        
        # Check if input is numpy array and convert to torch tensor + device
        organelle_seg, organelle_seg_is_numpy = self._check_input(organelle_seg, self.device)
        membrane_seg, membrane_seg_is_numpy = self._check_input(membrane_seg, self.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Step 1: Preprocess membrane mask
        logger.info("Preprocessing membrane segmentation")
        membrane_trimmed = self._trim_edges(membrane_seg)
        membrane_cleaned = self._remove_small_objects(
            membrane_trimmed, self.config.min_membrane_area
        ).bool().float()
        
        # Step 2: Filter organelles by membrane presence
        logger.info("Filtering organelles by membrane presence")
        membrane_z_presence = membrane_cleaned.sum(dim=(1, 2)) > 0
        organelle_filtered = organelle_seg * membrane_z_presence[:, None, None]
        
        # Step 3: Get organelle labels
        organelle_labels = torch.unique(organelle_filtered)
        organelle_labels = organelle_labels[organelle_labels > 0]
        
        if len(organelle_labels) == 0:
            logger.warning("No valid organelles found")
            empty_shape = organelle_seg.shape
            empty_tensor = torch.zeros(empty_shape, dtype=organelle_seg.dtype, device=self.device)
            return {
                'organelles': empty_tensor,
                'membranes': empty_tensor
            }
        
        logger.info(f"Found {len(organelle_labels)} organelles to process")
        
        # Process in batches on GPU
        organelle_relabeled = organelle_filtered.clone()
        organelle_relabeled[organelle_relabeled > 0] = (organelle_relabeled[organelle_relabeled > 0] + 1) * 2
        
        all_results = []
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(organelle_labels), batch_size), desc="Processing organelle batches"):
            batch_labels = organelle_labels[i:i + batch_size]
            batch_masks = []
            batch_even_labels = []
            
            # Create batch of organelle masks
            for label in batch_labels:
                even_label = (label + 1) * 2
                org_mask = organelle_relabeled.clone()
                org_mask[org_mask != even_label] = 0
                batch_masks.append(org_mask)
                batch_even_labels.append(even_label)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_masks)
            batch_labels_tensor = torch.tensor(batch_even_labels, device=self.device)
            
            # Process batch on GPU
            batch_results = self._process_organelle_batch(
                batch_tensor, membrane_cleaned, batch_labels_tensor
            )
            
            # Add non-None results
            all_results.extend([r for r in batch_results if r is not None])
        
        if not all_results:
            logger.warning("No valid organelle-membrane pairs found")
            empty_shape = organelle_seg.shape
            empty_tensor = torch.zeros(empty_shape, dtype=organelle_seg.dtype, device=self.device)
            return {
                'organelles': empty_tensor,
                'membranes': empty_tensor
            }
        
        # Combine results
        membrane_masks = torch.stack([r[0] for r in all_results])
        organelle_masks = torch.stack([r[1] for r in all_results])
        
        # Convert back from even-odd numbering
        organelle_instances = organelle_masks // 2
        membrane_instances = (membrane_masks + 1) // 2
        
        logger.info(f"Successfully processed {len(all_results)} organelle-membrane pairs")

        organelle_instances = self._return_results(organelle_instances, organelle_seg_is_numpy)
        membrane_instances = self._return_results(membrane_instances, membrane_seg_is_numpy)

        return { 'organelles': organelle_instances, 'membranes': membrane_instances }

    def convert_to_3d_labels(self, masks_4d: TensorLike) -> TensorLike:
        """Convert 4D instance masks to 3D label map, preserving existing labels."""
        
        # If no masks, return empty tensor
        if len(masks_4d) == 0:
            return torch.zeros(masks_4d.shape[1:], dtype=masks_4d.dtype) if isinstance(masks_4d, torch.Tensor) else np.zeros(masks_4d.shape[1:], dtype=masks_4d.dtype)
        
        if isinstance(masks_4d, np.ndarray):
            return self._convert3D_numpy(masks_4d)
        else:
            return self._convert3D_torch(masks_4d)

    def _convert3D_numpy(self, masks_4d: np.ndarray) -> np.ndarray:
        """Convert 4D instance masks to 3D label map, preserving existing labels."""
        output_3d = np.zeros(masks_4d.shape[1:], dtype=masks_4d.dtype)
        for i, mask in enumerate(masks_4d):
            output_3d[mask > 0] = mask[mask > 0]
        return output_3d

    def _convert3D_torch(self, masks_4d: torch.Tensor) -> torch.Tensor:
        """Convert 4D instance masks to 3D label map, preserving existing labels."""
        output_3d = torch.zeros(masks_4d.shape[1:], dtype=masks_4d.dtype, device=masks_4d.device)
        for i, mask in enumerate(masks_4d):
            output_3d[mask > 0] = mask[mask > 0]
        return output_3d
        
    def _check_input(self, input: TensorLike, device: torch.device) -> Tuple[torch.Tensor, bool]:
        """Check if input is numpy array and convert to torch tensor + device."""
        if isinstance(input, np.ndarray):
            input_is_numpy = True
            input = torch.from_numpy(input).to(device)
        else:
            input_is_numpy = False
            input = input.to(device)
        return input, input_is_numpy

    def _return_results(self, results: torch.Tensor, input_is_numpy: bool) -> TensorLike:
        """Return results as numpy array or torch tensor."""
        if input_is_numpy:
            return results.cpu().numpy()
        else:
            return results.cpu()

# Convenience function for simple usage
def filter_organelle_membrane_segmentation(
    organelle_seg: TensorLike,
    membrane_seg: TensorLike,
    ball_size: int = 15,
    min_membrane_area: int = 50000,
    min_organelle_area: int = 300000,
    batch_size: int = 8,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Convenience function for organelle-membrane filtering.
    
    Args:
        organelle_seg: 3D organelle instance segmentation
        membrane_seg: 3D binary membrane segmentation  
        ball_size: Size of morphological structuring element
        min_membrane_area: Minimum area for membrane components
        min_organelle_area: Minimum area for organelle components
        batch_size: Number of organelles to process simultaneously
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with filtered segmentations
    """
    config = FilteringConfig(
        ball_size=ball_size,
        min_membrane_area=min_membrane_area,
        min_organelle_area=min_organelle_area,
        batch_size=batch_size,
        **kwargs
    )
    
    filter_pipeline = OrganelleMembraneFilter(config)
    return filter_pipeline.filter_organelle_membrane_segmentation(organelle_seg, membrane_seg)