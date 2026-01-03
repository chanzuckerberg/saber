from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset
from scipy.ndimage import label
from functools import partial
import torch, zarr, os
from tqdm import tqdm
import numpy as np

class ZarrSegmentationDataset(Dataset):
    def __init__(self, zarr_path, mode='train', transform=None, min_area=500, 
                 negative_class_reduction=1, num_workers=None):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
        
        self.zarr_path = zarr_path
        self.zfile = zarr.open(zarr_path, mode='r')
        self.mode = mode
        self.min_area = min_area
        self.transform = transform
        self.negative_class_reduction = negative_class_reduction

        # Get all run IDs
        run_ids = [group[0] for group in self.zfile.groups()]
        
        # Build index in parallel
        self.sample_index = []
        
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, len(run_ids))
        
        if num_workers > 1 and len(run_ids) > 1:
            # Parallel indexing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                index_fn = partial(_index_run, 
                                   zarr_path=zarr_path,
                                   min_area=min_area,
                                   negative_class_reduction=negative_class_reduction)
                
                futures = {executor.submit(index_fn, run_id): run_id 
                          for run_id in run_ids}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), 
                                  total=len(futures), 
                                  desc="Indexing dataset (parallel)"):
                    samples = future.result()
                    self.sample_index.extend(samples)
        else:
            # Sequential fallback (for debugging or small datasets)
            for run_id in tqdm(run_ids, desc="Indexing dataset"):
                samples = _index_run(run_id, zarr_path, min_area, negative_class_reduction)
                self.sample_index.extend(samples)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        run_id, class_idx, component_idx, is_negative = self.sample_index[idx]
        
        # Load image and masks on-demand
        group = self.zfile[run_id]
        image = group['0'][:]
        
        # Get the specific mask
        if is_negative:
            mask_array = group['labels']['rejected'][:]
        else:
            mask_array = group['labels']['0'][:]
        
        # Extract the specific component
        labeled_mask, _ = label(mask_array[class_idx])
        component_mask = (labeled_mask == component_idx).astype(np.uint8)
        
        label_value = 0 if is_negative else class_idx
        
        # Apply transforms
        if self.transform:
            data = self.transform({'image': image, 'mask': component_mask})
            image = data['image']
            component_mask = data['mask']
        
        return {
            'image': image,
            'mask': component_mask,
            'label': torch.tensor(label_value, dtype=torch.long)
        }

def _index_run(run_id, zarr_path, min_area, negative_class_reduction):
    """Worker function to index a single run. Must be at module level for pickling."""
    zfile = zarr.open(zarr_path, mode='r')
    group = zfile[run_id]

    # Validate required structure
    if 'labels' not in group:
        return []

    labels = group['labels']
    samples = []
    
    # Index candidate masks
    if '0' in labels:
        candidate_masks = labels['0'][:]
        for class_idx, mask in enumerate(candidate_masks):
            if mask.max() > 0:
                labeled_mask, num_features = label(mask)
                for component_idx in range(1, num_features + 1):
                    component_mask = (labeled_mask == component_idx)
                    if component_mask.sum() > min_area:
                        samples.append((run_id, class_idx, component_idx, False))
    
    # Index rejected masks
    if 'rejected' in labels:
        rejected_masks = labels['rejected'][::negative_class_reduction]
        for class_idx, mask in enumerate(rejected_masks):
            if mask.max() > 0:
                labeled_mask, num_features = label(mask)
                for component_idx in range(1, num_features + 1):
                    component_mask = (labeled_mask == component_idx)
                    if component_mask.sum() > min_area:
                        samples.append((run_id, class_idx, component_idx, True))
    
    return samples