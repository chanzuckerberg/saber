from torch.utils.data import Dataset
from scipy.ndimage import label
import torch, zarr, os
from tqdm import tqdm
import numpy as np

class ZarrSegmentationDataset(Dataset):
    def __init__(self, zarr_path, mode='train', transform=None, min_area = 250):
        """
        Args:
            zarr_path (str): Path to the Zarr file.
            run_ids (list): List of run IDs to include.
            mode (str): 'train' for random sampling, 'inference' for deterministic order.
            transform (callable, optional): Optional transform to apply to the data.
            min_area (int, optional): Minimum area of masks to keep. Default is 100.
        """
        
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
        self.zfile = zarr.open(zarr_path, mode='r')
        self.mode = mode
        self.min_area = min_area
        self.transform = transform

        # Factor to reduce the number of negative samples
        negative_class_reduction = 1

        # Extract group names as a string array
        self.run_ids = [group[0] for group in self.zfile.groups()]

        # Preload all masks and labels for efficient access
        self.samples = []
        for run_id in tqdm(self.run_ids):
            group = self.zfile[run_id]
            image = group['image'][:]

            if 'masks' in group:
                # Process candidate masks
                candidate_masks = group['masks'][:] # [Nclass, Nx, Ny]
                self._process_masks(candidate_masks, image)
            else:
                continue
            
            # Check if "rejected_masks" exists before accessing
            if 'rejected_masks' in group:
                # Process rejected masks
                rejected_masks = group['rejected_masks'][::negative_class_reduction]
                self._process_masks(rejected_masks, image, is_negative_mask=True)  

    def _process_masks(self, masks, image, is_negative_mask = False):
        """
        Process masks by separating connected components and filtering out empty masks.
        
        Args:
            masks (numpy.ndarray): Array of masks to process.
            image (numpy.ndarray): Corresponding image data.
            label_value (int): Label to assign to the masks (1 for candidates, 0 for rejected).
        """
        for class_idx, mask in enumerate(masks):  # Iterate over each class dimension
            if mask.max() > 0:  # Ignore empty masks
                # Separate connected components
                labeled_mask, num_features = label(mask)
                for component_idx in range(1, num_features + 1):
                    component_mask = (labeled_mask == component_idx).astype(np.uint8)
                    if ( component_mask.max() > 0 ) and ( component_mask.sum() > self.min_area ):  # Ensure the component is non-empty
                        self.samples.append({
                            'image': image,
                            'mask': component_mask,
                            'label': 0 if is_negative_mask else class_idx + 1  # Assign labels properly
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image = self.samples[idx]['image']
        mask = self.samples[idx]['mask']
        label = self.samples[idx]['label']
        
        # Apply transforms
        if self.transform:
            data = self.transform({'image': image, 'mask': mask})
            image = data['image']
            mask = data['mask']        

        # Return as tensors
        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long)
        }