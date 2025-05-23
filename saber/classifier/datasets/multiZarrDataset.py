from saber.classifier.datasets import singleZarrDataset
from torch.utils.data import Dataset
from typing import List, Union
from tqdm import tqdm
import numpy as np
import torch, os

class MultiZarrDataset(Dataset):
    def __init__(self, zarr_paths, mode='train', transform=None):

        self.folder_path = zarr_paths
        self.transform = transform
        self.mode = mode

        # Handle zarr_paths input
        if isinstance(zarr_paths, str):
            # If zarr_paths is a folder, collect all Zarr files in the folder
            self.zarr_files: List[str] = [
                os.path.join(zarr_paths, f) for f in os.listdir(zarr_paths) if f.endswith('.zarr')
            ]
            # Check if there are any Zarr files
            if not self.zarr_files:
                raise ValueError(f"No Zarr files found in {zarr_paths}.")            
        elif isinstance(zarr_paths, list):
            # If zarr_paths is a list, assume it contains Zarr file paths
            self.zarr_files: List[str] = zarr_paths
        else:
            raise ValueError("zarr_paths must be either a folder path (str) or a list of file paths (List[str]).")   

        # Preload all Zarr datasets into memory with a progress bar
        self.datasets = [
            singleZarrDataset.ZarrSegmentationDataset(zarr_file, mode=mode, transform=transform)
            for zarr_file in tqdm(self.zarr_files, desc="Loading Zarr Datasets", unit="file")
        ]

        # Precompute file lengths and total number of samples
        self.file_lengths = [len(dataset) for dataset in self.datasets]
        self.num_samples = sum(self.file_lengths)       

    def __len__(self):
        """
        Returns the total number of samples in the current Zarr file.
        """
        return self.num_samples   

    def __getitem__(self, idx):
        """
        Fetch a sample by global index, mapping it to the correct dataset and local index.

        Args:
            idx (int): Global index of the sample.

        Returns:
            The corresponding sample from the appropriate dataset.
        """
        # Determine which dataset (Zarr file) the sample belongs to
        for file_idx, file_length in enumerate(self.file_lengths):
            if idx < file_length:
                # Fetch the sample from the corresponding dataset
                return self.datasets[file_idx][idx]
            idx -= file_length  # Adjust the index for the next dataset

        raise IndexError("Index out of range")
