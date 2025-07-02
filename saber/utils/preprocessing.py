from scipy.ndimage import uniform_filter
import numpy as np

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
