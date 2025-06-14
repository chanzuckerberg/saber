import saber.visualization.sam2 as viz
from scipy.ndimage import uniform_filter
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import cv2, torch, skimage
import numpy as np

def project_tomogram(vol, zSlice = None, deltaZ = None):
    """
    Projects a tomogram along the z-axis.
    
    Parameters:
    vol (np.ndarray): 3D tomogram array (z, y, x).
    zSlice (int, optional): Specific z-slice to project. If None, project along all z slices.
    deltaZ (int, optional): Thickness of slices to project. Used only if zSlice is specified. If None, project a single slice.

    Returns:
    np.ndarray: 2D projected tomogram.
    """    

    if zSlice is not None:
        # If deltaZ is specified, project over zSlice to zSlice + deltaZ
        if deltaZ is not None:
            zStart = int(max(zSlice - deltaZ, 0))
            zEnd = int(min(zSlice + deltaZ, vol.shape[0]))  # Ensure we don't exceed the volume size
            projection = np.mean(vol[zStart:zEnd,], axis=0)  # Sum over the specified slices
        else:
            # If deltaZ is not specified, project just a single z slice
            projection = vol[zSlice,]
    else:
        # If zSlice is None, project over the entire z-axis
        projection = np.mean(vol, axis=0)
        
    return projection

def cryodino_project_tomogram(vol, zSlice = None, deltaZ = None):
    """
    Projects a tomogram along the z-axis.
    
    Parameters:
    vol (np.ndarray): 3D tomogram array (z, y, x).
    zSlice (int, optional): Specific z-slice to project. If None, project along all z slices.
    deltaZ (int, optional): Thickness of slices to project. Used only if zSlice is specified. If None, project a single slice.

    Returns:
    np.ndarray: 2D projected tomogram.
    """    

    # Zero Mean, Unit Std Before Normalizing
    vol = (vol - vol.mean()) / vol.std()

    if zSlice is not None:
        # If deltaZ is specified, project over zSlice to zSlice + deltaZ
        if deltaZ is not None:
            zStart = int(max(zSlice - deltaZ, 0))
            zEnd = int(min(zSlice + deltaZ, vol.shape[0]))  # Ensure we don't exceed the volume size

            projection = np.mean(vol[zStart:zEnd,], axis=0)  # Sum over the specified slices

            projection = projection / projection.std()
        else:
            # If deltaZ is not specified, project just a single z slice
            projection = vol[zSlice,]
    else:
        # If zSlice is None, project over the entire z-axis
        projection = np.mean(vol, axis=0)

    # Clip to -4, 4 times STD
    projection = np.clip(projection, -4, 4)
        
    return projection

def contrast(image, std_cutoff=5):
    """
    Normalize the Input Data to [0,1]
    """
    image_mean = uniform_filter(image, size=500)
    image_sq = uniform_filter(image**2, size=500)
    image_var = np.clip(image_sq - image_mean**2, a_min=0, a_max=None)
    image_std = np.sqrt(image_var)
    image = (image - image_mean) / (image_std + 1e-8)

    return np.clip(image, -std_cutoff, std_cutoff)

def normalize(image, rgb = False):
    # Clip the Volume by ±5std
    if rgb:
        min_vals = image.min(axis=(0, 1), keepdims=True)
        max_vals = image.max(axis=(0, 1), keepdims=True)
    else:
        min_vals = image.min()
        max_vals = image.max()
    normalized = (image - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid div by zero
    return normalized

def get_available_devices(deviceID: int = None):
    """
    Get the available devices for the current system.
    """
    # Set device
    if deviceID is None:
        if torch.cuda.is_available():           device_type = 'cuda'
        elif torch.backends.mps.is_available(): device_type = "mps" 
        else:                                   device_type = "cpu" 
        device = torch.device(device_type)
    else:
        device = determine_device(deviceID)
    return device

def determine_device(deviceID: int = 0):
    """
    Determine the device for the given deviceID.
    """

    # First check if CUDA is available at all
    if torch.cuda.is_available():
        try:

            # Make sure the device ID is valid
            device_count = torch.cuda.device_count()
            if deviceID >= device_count:
                print(f"Warning: Requested CUDA device {deviceID} but only {device_count} devices available")
                print(f"Falling back to device 0")
                deviceID = 0

            # Safely try to get the device properties
            props = torch.cuda.get_device_properties(deviceID)
            device = torch.device(f"cuda:{deviceID}")
            
            # Enable TF32 for Ampere GPUs if available
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Only set up autocast after confirming device works
                # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
        except Exception as e:
            print(f"Error accessing CUDA device {deviceID}: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    else:
        device = torch.device("cpu")
        print("Using CPU for computation (no GPU available)")

    return device

def convert_segments_to_mask(video_segments, masks, mask_shape, nMasks):

    for frame_idx in list(video_segments):
        for jj in range(nMasks):
            resized_mask = skimage.transform.resize(
                video_segments[frame_idx][jj+1][0,], 
                (mask_shape[1], mask_shape[2]), anti_aliasing=False
            )
            mask_update = resized_mask > 0
            masks[frame_idx,:,:][mask_update] = jj + 1    

    return masks

