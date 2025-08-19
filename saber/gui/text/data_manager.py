"""
Data management for SAM2-ET text annotations using run-level storage.
Each run stores its own annotations as attributes.
"""

import os
import json
import zarr
import numpy as np
from typing import Dict, Any
from datetime import datetime


class TextAnnotationDataManager:
    """Manages all data operations for text annotations with run-level storage."""
    
    def __init__(self, zarr_path: str, save_path: str):
        self.zarr_path = zarr_path
        self.save_path = save_path
        
        # Initialize storage (now only for current session - data is per-run)
        self.global_descriptions = {}  # run_id -> global description text
        self.segmentation_descriptions = {}  # run_id -> {segmentation_id -> description}

        self.session_masks_by_run = {}      # run_id -> List[np.ndarray]
        self.session_accepted_by_run = {}   # run_id -> Set[int]
        
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
    
    def load_run_annotations(self, run_id: str, hashtag_manager):
        """Load annotations for a specific run from its attributes."""
        print(f"🔍 Loading annotations for run: {run_id}")
        
        try:
            # Clear previous data for this run
            if run_id in self.global_descriptions:
                del self.global_descriptions[run_id]
            if run_id in self.segmentation_descriptions:
                del self.segmentation_descriptions[run_id]
            
            # Try input zarr first (for initial data)
            success = self._load_run_from_zarr(self.root, run_id, hashtag_manager, "input")
            
            # Try output zarr (for user modifications) - this overwrites input data
            if self.save_path and os.path.exists(self.save_path):
                save_root = zarr.open(self.save_path, mode='r')
                if run_id in save_root:
                    success = self._load_run_from_zarr(save_root, run_id, hashtag_manager, "output") or success
            
            if success:
                print(f"✅ Loaded annotations for {run_id}")
                if run_id in self.segmentation_descriptions:
                    print(f"   {len(self.segmentation_descriptions[run_id])} segmentation descriptions")
                if run_id in self.global_descriptions:
                    print(f"   Global description: '{self.global_descriptions[run_id]}'")
            else:
                print(f"⚠️ No annotations found for {run_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error loading annotations for {run_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_run_from_zarr(self, zarr_root, run_id: str, hashtag_manager, source_name: str):
        """Load run annotations from a specific zarr file."""
        try:
            if run_id not in zarr_root:
                return False
                
            run_group = zarr_root[run_id]
            if 'text_annotations' not in run_group.attrs:
                print(f"   No text_annotations found in {source_name} zarr for {run_id}")
                return False
            
            run_data = json.loads(run_group.attrs['text_annotations'])
            print(f"   ✅ Found text_annotations in {source_name} zarr for {run_id}")
            
            # Load global description
            global_desc = run_data.get('global_description', '')
            if global_desc:
                self.global_descriptions[run_id] = global_desc
            
            # Load segmentation descriptions
            seg_descriptions = run_data.get('segmentation_descriptions', {})
            if seg_descriptions:
                self.segmentation_descriptions[run_id] = seg_descriptions
            
            # Load hashtag data for this run
            hashtag_data = run_data.get('hashtag_data', {})
            if hashtag_data:
                # Clear existing hashtag data for this run
                hashtag_manager.clear_run_hashtags(run_id)
                
                # Add hashtags for this run
                for hashtag, indices in hashtag_data.items():
                    if hashtag not in hashtag_manager.hashtag_data:
                        hashtag_manager.hashtag_data[hashtag] = {}
                    hashtag_manager.hashtag_data[hashtag][run_id] = indices
            
            print(f"      Loaded {len(seg_descriptions)} descriptions, {len(hashtag_data)} hashtags")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading from {source_name} zarr: {e}")
            return False

    def _load_saved_masks(self, run_id: str):
        """Return (accepted, rejected) dicts: {seg_index:int -> mask:np.ndarray}."""
        accepted, rejected = {}, {}
        if not (self.save_path and os.path.exists(self.save_path)):
            return accepted, rejected

        try:
            root = zarr.open(self.save_path, mode='r')
            if run_id not in root:
                return accepted, rejected

            g = root[run_id]
            # accepted masks live under run_id/masks/segmentation_{i}/mask
            if 'masks' in g:
                mg = g['masks']
                for key in mg.keys():
                    seg = mg[key]
                    if 'mask' in seg:
                        idx = int(seg.attrs.get('segmentation_id', key.split('_')[-1]))
                        accepted[idx] = seg['mask'][:]

            # rejected masks live under run_id/rejected_masks/rejected_{i}
            if 'rejected_masks' in g:
                rg = g['rejected_masks']
                for key in rg.keys():
                    idx = int(key.split('_')[-1])
                    rejected[idx] = rg[key][:]
        except Exception as e:
            print(f"[_load_saved_masks] error for {run_id}: {e}")

        return accepted, rejected
    
    def read_augmented_data(self, run_id: str):
        """
        Read the original image+masks from the input zarr and overlay anything found
        in the save zarr (both accepted and rejected), returning:
        base_image, masks_list, accepted_indices_set
        """
        base_image, masks = self.read_data(run_id)  # keeps your transpose logic
        accepted_d, rejected_d = self._load_saved_masks(run_id)

        # start with the original list
        masks_list = [m.astype(np.float32) for m in masks]

        H, W = base_image.shape  # viewer convention: mask[x, y] with shape (H, W)

        def ensure_len(n):
            while len(masks_list) < n:
                masks_list.append(np.zeros((H, W), dtype=np.float32))

        # place accepted masks at their saved indices (overwrite or extend)
        for idx, m in accepted_d.items():
            ensure_len(idx + 1)  # ensure list is at least idx+1 long
            if idx < len(masks_list):
                masks_list[idx] = m.astype(np.float32)
            else:
                # This shouldn't happen with ensure_len, but just in case
                while len(masks_list) <= idx:
                    masks_list.append(np.zeros((H, W), dtype=np.float32))
                masks_list[idx] = m.astype(np.float32)

        # place rejected masks similarly (so they show on the LEFT on reload)
        for idx, m in rejected_d.items():
            ensure_len(idx + 1)  # ensure list is at least idx+1 long
            if idx < len(masks_list):
                masks_list[idx] = m.astype(np.float32)
            else:
                # This shouldn't happen with ensure_len, but just in case
                while len(masks_list) <= idx:
                    masks_list.append(np.zeros((H, W), dtype=np.float32))
                masks_list[idx] = m.astype(np.float32)

        accepted_indices = set(accepted_d.keys())
        return base_image, masks_list, accepted_indices

    def stash_session_state(self, run_id: str, viewer):
        """Copy current viewer masks/accepted into an in-memory cache for this run."""
        if not run_id or viewer is None:
            return
        # Deep-copy masks so later edits don't mutate the cache
        self.session_masks_by_run[run_id] = [m.copy() for m in viewer.masks]
        self.session_accepted_by_run[run_id] = set(int(i) for i in getattr(viewer, 'accepted_masks', set()))

    def clear_session_state(self, run_id: str):
        """Optionally clear the cache for a run after saving, if you want."""
        self.session_masks_by_run.pop(run_id, None)
        self.session_accepted_by_run.pop(run_id, None)

    def read_with_session_fallback(self, run_id: str):
        """
        If we have a session cache for this run, return it.
        Otherwise, return the saved-augmented data (accepted+rejected merged) from disk.
        """
        base_image, _base_masks = self.read_data(run_id)
        if run_id in self.session_masks_by_run:
            masks_list = self.session_masks_by_run[run_id]
            accepted = self.session_accepted_by_run.get(run_id, set())
            return base_image, masks_list, accepted
        # Fallback: what's on disk (source + saved adds)
        return self.read_augmented_data(run_id)
    
    def save_run_annotations(self, run_id: str, hashtag_manager):
        """Save annotations for a specific run to its attributes."""
        if not self.save_path:
            print("Warning: No save path specified.")
            return False
        
        try:
            zarr_root = zarr.open(self.save_path, mode='a')
            
            # Ensure run group exists
            if run_id not in zarr_root:
                print(f"Warning: Run {run_id} not found in save zarr. Cannot save annotations.")
                return False
            
            # Build run-specific annotation data
            run_annotations = {
                'global_description': self.global_descriptions.get(run_id, ''),
                'segmentation_descriptions': self.segmentation_descriptions.get(run_id, {}),
                'hashtag_data': {},
                'last_modified': datetime.now().isoformat()
            }
            
            # Extract hashtag data for this run only
            for hashtag, runs_data in hashtag_manager.hashtag_data.items():
                if run_id in runs_data:
                    run_annotations['hashtag_data'][hashtag] = runs_data[run_id]
            
            # Save hashtag colors
            run_annotations['hashtag_colors'] = hashtag_manager.hashtag_colors
            
            # Save to run attributes
            zarr_root[run_id].attrs['text_annotations'] = json.dumps(run_annotations)
            
            print(f"✅ Saved run-level annotations for {run_id}")
            print(f"   Global desc: {'Yes' if run_annotations['global_description'] else 'No'}")
            print(f"   Segmentation descs: {len(run_annotations['segmentation_descriptions'])}")
            print(f"   Hashtags: {len(run_annotations['hashtag_data'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save run annotations for {run_id}: {e}")
            return False
    
    def save_text_data(self, hashtag_manager):
        """Save text data - now just delegates to current run."""
        # This method is kept for compatibility but now works per-run
        print("💾 Saving text data (run-level)...")
        success_count = 0
        
        # Save annotations for all runs that have been modified
        for run_id in set(list(self.global_descriptions.keys()) + list(self.segmentation_descriptions.keys())):
            if self.save_run_annotations(run_id, hashtag_manager):
                success_count += 1
        
        print(f"✅ Saved annotations for {success_count} runs")
        return success_count > 0
    
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

        # Always rebuild groups so no stale accepted/rejected remain
        if 'masks' in segmentation_group:
            del segmentation_group['masks']
        masks_group = segmentation_group.create_group('masks')

        if 'rejected_masks' in segmentation_group:
            del segmentation_group['rejected_masks']
        rejected_group = segmentation_group.create_group('rejected_masks')

        if total_masks == 0:
            print(f"No masks to save for run ID {run_id}")
            return

        # Current acceptance set
        accepted_indices = set(getattr(segmentation_viewer, 'accepted_masks', set()))
        all_indices = set(range(total_masks))
        rejected_indices = all_indices - accepted_indices

        # ---- Write accepted masks + attrs ----
        for i in sorted(accepted_indices):
            if 0 <= i < total_masks:
                seg_group = masks_group.require_group(f'segmentation_{i}')
                seg_group['mask'] = segmentation_viewer.masks[i].astype(np.uint8)

                description = self.segmentation_descriptions.get(run_id, {}).get(str(i), '')
                hashtags = self._extract_hashtags(description)
                bbox = self._get_mask_bbox(segmentation_viewer.masks[i])

                seg_group.attrs['description'] = description
                seg_group.attrs['hashtags'] = json.dumps(hashtags)
                seg_group.attrs['bbox'] = bbox
                seg_group.attrs['area'] = int(np.sum(segmentation_viewer.masks[i] > 0))
                seg_group.attrs['segmentation_id'] = int(i)

        print(f"Saved {len(accepted_indices)} mask+description pairs for runID: {run_id}")

        # ---- Write rejected masks (no attrs needed) ----
        for idx in sorted(rejected_indices):
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
        """Save current text to memory for a specific run."""
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
    
    def debug_zarr_contents(self):
        """Debug method to see what's actually in the zarr files."""
        print(f"\n🔍 DEBUG ZARR CONTENTS (Run-Level):")
        
        # Check input zarr
        print(f"Input zarr ({self.zarr_path}):")
        print(f"  Run IDs: {self.run_ids}")
        
        # Check a few runs for annotations
        for run_id in self.run_ids[:3]:  # Check first 3 runs
            if 'text_annotations' in self.root[run_id].attrs:
                try:
                    run_data = json.loads(self.root[run_id].attrs['text_annotations'])
                    print(f"  ✅ {run_id}: found text_annotations")
                    print(f"     Keys: {list(run_data.keys())}")
                    
                    if 'segmentation_descriptions' in run_data:
                        seg_count = len(run_data['segmentation_descriptions'])
                        print(f"     Segmentation descriptions: {seg_count}")
                    
                    if 'hashtag_data' in run_data:
                        hashtag_count = len(run_data['hashtag_data'])
                        print(f"     Hashtags: {hashtag_count} - {list(run_data['hashtag_data'].keys())}")
                        
                except Exception as e:
                    print(f"  ❌ Error reading {run_id}: {e}")
            else:
                print(f"  ❌ {run_id}: no text_annotations")
        
        # Check global index
        if 'global_index' in self.root.attrs:
            try:
                global_index = json.loads(self.root.attrs['global_index'])
                print(f"  ✅ Found global_index:")
                print(f"     Runs processed: {len(global_index.get('runs_processed', []))}")
                print(f"     Total components: {global_index.get('total_components', 0)}")
                print(f"     Hashtags: {list(global_index.get('hashtag_summary', {}).keys())}")
            except Exception as e:
                print(f"  ❌ Error reading global_index: {e}")
        else:
            print(f"  ❌ No global_index found")
        
        # Check output zarr if it exists
        print(f"\nOutput zarr ({self.save_path}):")
        if os.path.exists(self.save_path):
            try:
                output_root = zarr.open(self.save_path, mode='r')
                output_run_ids = list(output_root.keys())
                print(f"  Run IDs: {output_run_ids}")
                
                # Check first run for annotations
                if output_run_ids:
                    first_run = output_run_ids[0]
                    if 'text_annotations' in output_root[first_run].attrs:
                        output_data = json.loads(output_root[first_run].attrs['text_annotations'])
                        # print(f"  ✅ {first_run}: found text_annotations")
                        # print(f"     Keys: {list(output_data.keys())}")
                    else:
                        print(f"  ❌ {first_run}: no text_annotations")
            except Exception as e:
                print(f"  ❌ Error reading output zarr: {e}")
        else:
            print(f"  ❌ Output zarr does not exist")

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