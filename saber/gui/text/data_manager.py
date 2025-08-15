"""
Data management for SAM2-ET text annotations.
Handles loading/saving zarr data, text annotations, and mask descriptions.
"""

import os
import json
import zarr
import numpy as np
from typing import Dict, Any
from datetime import datetime


class TextAnnotationDataManager:
    """Manages all data operations for text annotations."""
    
    def __init__(self, zarr_path: str, save_path: str):
        self.zarr_path = zarr_path
        self.save_path = save_path
        
        # Initialize storage
        self.global_descriptions = {}  # run_id -> global description text
        self.segmentation_descriptions = {}  # run_id -> {segmentation_id -> description}
        
        # Load zarr data
        self.load_zarr_data()
        
    def load_zarr_data(self):
        """Load zarr data and run IDs."""
        if os.path.exists(self.zarr_path):
            self.root = zarr.open(self.zarr_path, mode='r')
        else:
            raise FileNotFoundError(f"Zarr file {self.zarr_path} does not exist.")
        self.run_ids = list(self.root.keys())
        self.good_run_ids = []
    
    def read_data(self, run_id: str):
        """Read image and mask data for a run ID."""
        base_image = self.root[run_id]['image'][:]
        try:
            masks = self.root[run_id]['labels'][:]
        except:
            masks = self.root[run_id]['masks'][:]

        (nx, ny) = base_image.shape
        if nx < ny:
            base_image = base_image.T
            masks = np.swapaxes(masks, 1, 2)

        return base_image, masks
    
    def load_existing_text_data(self, hashtag_manager):
        """Load existing text data from save file."""
        if self.save_path and os.path.exists(self.save_path):
            try:
                save_root = zarr.open(self.save_path, mode='r')
                if 'text_annotations' in save_root.attrs:
                    text_data = json.loads(save_root.attrs['text_annotations'])
                    self.global_descriptions = text_data.get('global_descriptions', {})
                    self.segmentation_descriptions = text_data.get('segmentation_descriptions', {})
                    hashtag_manager.load_data_from_save(text_data)
                    print("Loaded existing text annotations.")
            except Exception as e:
                print(f"Could not load existing text data: {e}")
    
    def save_text_data(self, hashtag_manager):
        """Save text data to zarr file."""
        if not self.save_path:
            print("Warning: No save path specified.")
            return False
        
        try:
            zarr_root = zarr.open(self.save_path, mode='a')
            text_data = {
                'global_descriptions': self.global_descriptions,
                'segmentation_descriptions': self.segmentation_descriptions,
                **hashtag_manager.get_data_for_save()
            }
            zarr_root.attrs['text_annotations'] = json.dumps(text_data)
            print("Text annotations saved successfully.")
            return True
        except Exception as e:
            print(f"Failed to save text data: {e}")
            return False
    
    def save_masks_data(self, segmentation_viewer, run_id: str):
        """Save mask data to zarr file."""
        zarr_root = zarr.open(self.save_path, mode='a')
        
        if run_id in zarr_root:
            print(f"\nWarning: Overwriting existing group {run_id}")
        
        segmentation_group = zarr_root.require_group(run_id)
        current_image = segmentation_viewer.left_base_img_item.image
        segmentation_group['image'] = current_image
        
        try:
            self.save_masks_to_zarr(segmentation_group, run_id, segmentation_viewer)
            zarr_root.attrs['good_run_ids'] = self.good_run_ids
            return True
        except Exception as e:
            print(f"Error saving masks for run ID {run_id}: {e}")
            return False
    
    def save_masks_to_zarr(self, segmentation_group, run_id: str, segmentation_viewer):
        """Save masks with integrated descriptions as structured data."""
        total_masks = len(segmentation_viewer.masks)
        
        if total_masks == 0:
            print(f"No masks to save for run ID {run_id}")
            return
        
        # Get accepted masks
        accepted_indices = getattr(segmentation_viewer, 'accepted_masks', set())
        
        if not accepted_indices:
            print(f"No accepted masks to save for run ID '{run_id}'.")
            return
        
        # Save accepted masks with descriptions
        masks_group = segmentation_group.require_group('masks')
        
        for i in accepted_indices:
            if 0 <= i < total_masks:
                seg_group = masks_group.require_group(f'segmentation_{i}')
                
                # Store mask array
                seg_group['mask'] = segmentation_viewer.masks[i].astype(np.uint8)
                
                # Store description and metadata as attributes
                description = self.segmentation_descriptions.get(run_id, {}).get(str(i), '')
                hashtags = self._extract_hashtags(description)
                bbox = self._get_mask_bbox(segmentation_viewer.masks[i])
                
                seg_group.attrs['description'] = description
                seg_group.attrs['hashtags'] = json.dumps(hashtags)
                seg_group.attrs['bbox'] = bbox
                seg_group.attrs['area'] = int(np.sum(segmentation_viewer.masks[i] > 0))
                seg_group.attrs['segmentation_id'] = i
        
        print(f'Saved {len(accepted_indices)} mask+description pairs for runID: {run_id}')
        
        # Save rejected masks (simple, no descriptions)
        all_indices = set(range(total_masks))
        rejected_indices = all_indices - accepted_indices
        
        if rejected_indices:
            rejected_group = segmentation_group.require_group('rejected_masks')
            for idx in rejected_indices:
                if 0 <= idx < total_masks:
                    rejected_group[f'rejected_{idx}'] = segmentation_viewer.masks[idx].astype(np.uint8)
    
    def load_masks_with_descriptions(self, run_id: str):
        """Load masks with their descriptions as a unified dictionary."""
        if not os.path.exists(self.save_path):
            return {}
        
        try:
            zarr_root = zarr.open(self.save_path, mode='r')
            if run_id not in zarr_root or 'masks' not in zarr_root[run_id]:
                return {}
            
            masks_group = zarr_root[run_id]['masks']
            result = {}
            
            for seg_key in masks_group.keys():
                seg_group = masks_group[seg_key]
                if 'mask' in seg_group:
                    result[seg_key] = {
                        'mask': seg_group['mask'][:],
                        'description': seg_group.attrs.get('description', ''),
                        'hashtags': json.loads(seg_group.attrs.get('hashtags', '[]')),
                        'bbox': list(seg_group.attrs.get('bbox', [0, 0, 0, 0])),
                        'area': seg_group.attrs.get('area', 0),
                        'segmentation_id': seg_group.attrs.get('segmentation_id', 0)
                    }
            
            return result
            
        except Exception as e:
            print(f"Error loading masks with descriptions for {run_id}: {e}")
            return {}
    
    def save_text_to_memory(self, run_id: str, global_text: str, selected_id: int = None, seg_text: str = ""):
        """Save current text to memory."""
        if not run_id:
            return
        
        # Save global description
        global_text = global_text.strip()
        if global_text:
            self.global_descriptions[run_id] = global_text
        elif run_id in self.global_descriptions:
            del self.global_descriptions[run_id]
        
        # Save segmentation description
        if selected_id is not None:
            seg_text = seg_text.strip()
            if run_id not in self.segmentation_descriptions:
                self.segmentation_descriptions[run_id] = {}
            
            seg_key = str(selected_id)
            if seg_text:
                self.segmentation_descriptions[run_id][seg_key] = seg_text
            elif seg_key in self.segmentation_descriptions[run_id]:
                del self.segmentation_descriptions[run_id][seg_key]
            
            # Clean up empty entries
            if not self.segmentation_descriptions[run_id]:
                del self.segmentation_descriptions[run_id]
    
    def _get_mask_bbox(self, mask: np.ndarray):
        """Get bounding box for a mask."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return [0, 0, 0, 0]
        
        return [
            int(cols.min()), int(rows.min()),  # x_min, y_min
            int(cols.max()), int(rows.max())   # x_max, y_max
        ]
    
    def _extract_hashtags(self, text: str):
        """Extract hashtags from text - simple version."""
        import re
        hashtag_pattern = r'#\w+'
        return list(set(re.findall(hashtag_pattern, text.lower())))
