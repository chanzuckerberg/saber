from __future__ import annotations

import torch
import torch.nn.functional as F
import kornia.morphology as km
import kornia.contrib
import numpy as np
import logging
from typing import Union, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class FilteringConfig:
    """Configuration for organelle-membrane filtering pipeline."""
    ball_size: int = 15
    min_membrane_area: int = 50000
    min_organelle_area: int = 300000
    edge_trim_z: int = 20
    edge_trim_xy: int = 15
    dilate_final_mask: bool = False
    dilate_size: Optional[int] = None
    min_roi_relative_size: float = 0.15


class OrganelleMembraneFilter:
    """
    Filters organelle and membrane segmentations to eliminate false positives
    and retain only membrane segmentations on organelle surfaces.
    
    The pipeline:
    1. Preprocesses masks (edge trimming, size filtering)
    2. Associates organelles with membranes
    3. Uses morphological operations to clean up segmentations
    4. Converts binary membrane masks to instance segmentations per organelle
    """
    
    def __init__(self, config: FilteringConfig = None):
        self.config = config or FilteringConfig()
        self._ball_kernel_cache = {}
    
    def _get_ball_kernel(self, radius: int, device: torch.device) -> torch.Tensor:
        """Get cached ball kernel or create new one."""
        cache_key = (radius, device)
        if cache_key not in self._ball_kernel_cache:
            self._ball_kernel_cache[cache_key] = self._create_ball_kernel(radius, device)
        return self._ball_kernel_cache[cache_key]
    
    @staticmethod
    def _create_ball_kernel(radius: int, device: torch.device) -> torch.Tensor:
        """Create 3D ball-shaped structuring element."""
        size = 2 * radius + 1
        center = radius
        
        # Create coordinate grids
        z, y, x = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device), 
            torch.arange(size, device=device),
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
        trimmed[z_trim:-z_trim] = mask[z_trim:-z_trim]
        
        # Trim XY edges
        mask = trimmed
        trimmed = torch.zeros_like(mask)
        xy_trim = self.config.edge_trim_xy
        trimmed[:, xy_trim:-xy_trim, xy_trim:-xy_trim] = mask[:, xy_trim:-xy_trim, xy_trim:-xy_trim]
        
        return trimmed
    
    def _remove_small_objects(self, mask: torch.Tensor, min_size: int) -> torch.Tensor:
        """Remove connected components smaller than min_size."""
        if mask.sum() == 0:
            return mask
        
        # Get connected components
        binary_mask = (mask > 0).float()
        labels = kornia.contrib.connected_components(binary_mask.unsqueeze(0)).squeeze(0)
        
        # Find components to keep
        unique_labels, counts = torch.unique(labels, return_counts=True)
        keep_labels = unique_labels[(unique_labels != 0) & (counts >= min_size)]
        
        if len(keep_labels) == 0:
            return torch.zeros_like(mask)
        
        # Create mask for keeping only large objects
        keep_mask = torch.isin(labels, keep_labels)
        return mask * keep_mask
    
    def _get_largest_component(self, mask: torch.Tensor) -> torch.Tensor:
        """Keep only the largest connected component."""
        if mask.sum() == 0:
            return mask
        
        binary_mask = (mask > 0).float()
        labels = kornia.contrib.connected_components(binary_mask.unsqueeze(0)).squeeze(0)
        
        # Find largest component
        unique_labels, counts = torch.unique(labels, return_counts=True)
        non_zero_mask = unique_labels != 0
        
        if not non_zero_mask.any():
            return mask
        
        largest_label = unique_labels[non_zero_mask][counts[non_zero_mask].argmax()]
        largest_mask = labels == largest_label
        
        return mask * largest_mask
    
    def _get_organelle_roi(self, organelle_mask: torch.Tensor, pad: int) -> Optional[Tuple[int, ...]]:
        """Get bounding box ROI for organelle with padding."""
        nonzero_indices = torch.nonzero(organelle_mask, as_tuple=False)
        if len(nonzero_indices) == 0:
            return None
        
        mins = nonzero_indices.min(dim=0)[0]
        maxs = nonzero_indices.max(dim=0)[0] + 1  # Make inclusive
        
        # Check minimum size requirement
        sizes = maxs - mins
        shape = torch.tensor(organelle_mask.shape)
        min_sizes = self.config.min_roi_relative_size * shape
        
        if (sizes < min_sizes).any():
            return None
        
        # Apply padding and clamp to bounds
        mins = torch.clamp(mins - pad, 0)
        maxs = torch.clamp(maxs + pad, max=shape)
        
        return tuple(mins.tolist() + maxs.tolist())
    
    def _morphological_opening(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Perform morphological opening (erosion followed by dilation)."""
        # Add batch/channel dimensions for kornia
        image_4d = image.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Erosion followed by dilation
        eroded = km.erosion(image_4d, kernel_4d)
        opened = km.dilation(eroded, kernel_4d)
        
        return opened.squeeze(0).squeeze(0)
    
    def _process_organelle_membrane_pair(
        self, 
        organelle_label: int,
        organelle_mask: torch.Tensor, 
        membrane_mask: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Process a single organelle-membrane pair.
        
        Returns:
            Tuple of (combined_mask, organelle_mask, membrane_instance_mask) or None
        """
        logger.debug(f"Processing organelle label {organelle_label}")
        
        # Get ROI for this organelle
        ball_size = self.config.ball_size
        roi = self._get_organelle_roi(organelle_mask, pad=ball_size // 2)
        if roi is None:
            logger.debug(f"Organelle {organelle_label} too small, skipping")
            return None
        
        minz, miny, minx, maxz, maxy, maxx = roi
        
        # Extract ROI regions
        org_roi = organelle_mask[minz:maxz, miny:maxy, minx:maxx]
        mem_roi = membrane_mask[minz:maxz, miny:maxy, minx:maxx]
        
        # Combine organelle and membrane: subtract membrane from organelle
        # This creates a mask where organelle=positive, membrane=negative
        combined_roi = org_roi.to(torch.int32) - mem_roi.to(torch.int32)
        combined_roi = torch.where(combined_roi != 0, torch.maximum(combined_roi, torch.tensor(1)), combined_roi)
        
        # Apply morphological opening to clean up the combined mask
        ball_kernel = self._get_ball_kernel(ball_size, organelle_mask.device)
        cleaned_roi = self._morphological_opening(combined_roi.float(), ball_kernel)
        
        # Optional dilation
        if self.config.dilate_final_mask:
            dilate_size = self.config.dilate_size or ball_size // 2
            dilate_kernel = self._get_ball_kernel(dilate_size, organelle_mask.device)
            dilate_4d = cleaned_roi.unsqueeze(0).unsqueeze(0)
            kernel_4d = dilate_kernel.unsqueeze(0).unsqueeze(0)
            cleaned_roi = km.dilation(dilate_4d, kernel_4d).squeeze(0).squeeze(0)
        
        if cleaned_roi.sum() == 0:
            logger.debug(f"Organelle {organelle_label} eliminated by morphological operations")
            return None
        
        # Keep only largest connected component
        cleaned_roi = self._get_largest_component(cleaned_roi)
        
        # Create full-size output masks
        combined_full = torch.zeros_like(organelle_mask)
        combined_full[minz:maxz, miny:maxy, minx:maxx] = cleaned_roi
        
        # Create cleaned organelle mask (intersection with cleaned combined mask)
        organelle_cleaned_roi = org_roi * (cleaned_roi > 0)
        organelle_cleaned_roi = self._get_largest_component(organelle_cleaned_roi)
        organelle_full = torch.zeros_like(organelle_mask)
        organelle_full[minz:maxz, miny:maxy, minx:maxx] = organelle_cleaned_roi
        
        # Create membrane instance mask (labeled with organelle_label - 1)
        membrane_instance_roi = mem_roi * (cleaned_roi > 0)
        membrane_instance_full = torch.zeros_like(organelle_mask)
        membrane_instance_full[minz:maxz, miny:maxy, minx:maxx] = membrane_instance_roi
        membrane_instance_full[membrane_instance_full > 0] = organelle_label - 1
        
        return combined_full, organelle_full, membrane_instance_full
    
    def filter_organelle_membrane_segmentation(
        self,
        organelle_seg: TensorLike,
        membrane_seg: TensorLike
    ) -> Dict[str, torch.Tensor]:
        """
        Main filtering pipeline.
        
        Args:
            organelle_seg: 3D tensor with organelle instance labels
            membrane_seg: 3D binary tensor with membrane segmentation
            
        Returns:
            Dictionary with keys:
            - 'organelle_instances': Cleaned organelle instance segmentation
            - 'membrane_instances': Membrane converted to instance segmentation per organelle
            - 'combined_masks': Combined organelle+membrane masks per organelle
        """
        logger.info("Starting organelle-membrane filtering pipeline")
        
        # Convert inputs to tensors
        organelle_seg = torch.as_tensor(organelle_seg)
        membrane_seg = torch.as_tensor(membrane_seg)
        
        # Step 1: Preprocess membrane mask
        logger.info("Preprocessing membrane segmentation")
        membrane_trimmed = self._trim_edges(membrane_seg)
        membrane_cleaned = self._remove_small_objects(
            membrane_trimmed, self.config.min_membrane_area
        ).bool().float()  # Ensure binary
        
        # Step 2: Filter organelles by membrane presence
        logger.info("Filtering organelles by membrane presence")
        membrane_z_presence = membrane_cleaned.sum(dim=(1, 2)) > 0
        organelle_filtered = organelle_seg * membrane_z_presence[:, None, None]
        
        # Step 3: Remove small organelles
        organelle_cleaned = self._remove_small_objects(
            organelle_filtered, self.config.min_organelle_area
        )
        
        # Step 4: Process each organelle individually
        logger.info("Processing individual organelles")
        organelle_labels = torch.unique(organelle_cleaned)
        organelle_labels = organelle_labels[organelle_labels > 0]
        
        logger.info(f"Found {len(organelle_labels)} organelles to process")
        
        # Apply even-odd numbering for internal processing
        organelle_relabeled = organelle_cleaned.clone()
        organelle_relabeled[organelle_relabeled > 0] = (organelle_relabeled[organelle_relabeled > 0] + 1) * 2
        
        # Process each organelle
        results = []
        for label in organelle_labels:
            # Get individual organelle mask
            even_label = (label + 1) * 2  # Convert to even-odd numbering
            org_mask = organelle_relabeled.clone()
            org_mask[org_mask != even_label] = 0
            
            result = self._process_organelle_membrane_pair(
                even_label, org_mask, membrane_cleaned
            )
            if result is not None:
                results.append(result)
        
        if not results:
            logger.warning("No valid organelle-membrane pairs found")
            empty_shape = organelle_seg.shape
            empty_tensor = torch.zeros(empty_shape, dtype=organelle_seg.dtype, device=organelle_seg.device)
            return {
                'organelle_instances': empty_tensor,
                'membrane_instances': empty_tensor,
                'combined_masks': empty_tensor
            }
        
        # Combine results
        combined_masks = torch.stack([r[0] for r in results])
        organelle_masks = torch.stack([r[1] for r in results])
        membrane_masks = torch.stack([r[2] for r in results])
        
        # Convert back from even-odd numbering
        organelle_instances = organelle_masks // 2
        membrane_instances = (membrane_masks + 1) // 2
        
        logger.info(f"Successfully processed {len(results)} organelle-membrane pairs")
        
        return {
            'organelle_instances': organelle_instances,
            'membrane_instances': membrane_instances, 
            'combined_masks': combined_masks
        }


# Convenience function for simple usage
def filter_organelle_membrane_segmentation(
    organelle_seg: TensorLike,
    membrane_seg: TensorLike,
    ball_size: int = 15,
    min_membrane_area: int = 50000,
    min_organelle_area: int = 300000,
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
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with filtered segmentations
    """
    config = FilteringConfig(
        ball_size=ball_size,
        min_membrane_area=min_membrane_area,
        min_organelle_area=min_organelle_area,
        **kwargs
    )
    
    filter_pipeline = OrganelleMembraneFilter(config)
    return filter_pipeline.filter_organelle_membrane_segmentation(organelle_seg, membrane_seg)


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    organelle_seg = torch.randint(0, 5, (50, 100, 100))
    membrane_seg = torch.randint(0, 2, (50, 100, 100)).float()
    
    results = filter_organelle_membrane_segmentation(
        organelle_seg, 
        membrane_seg,
        ball_size=5,
        min_membrane_area=100,
        min_organelle_area=500
    )
    
    print(f"Organelle instances shape: {results['organelle_instances'].shape}")
    print(f"Membrane instances shape: {results['membrane_instances'].shape}")
    print(f"Combined masks shape: {results['combined_masks'].shape}")