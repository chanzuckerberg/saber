import torch.nn.functional as F
from saber.utils import io
import torch.nn as nn
import numpy as np
import torch, gc

def make_gaussian_kernel(sigma):
    ks = round(sigma * 3)
    ks = max(ks, 3)
    # Ensure the kernel size is odd
    ks += 1 - ks % 2  
    ts = torch.linspace(-ks / 2, ks / 2, ks)
    gauss = torch.exp(-(ts / sigma) ** 2 / 2)
    kernel = gauss / gauss.sum()
    return kernel

def gaussian_smoothing(input_tensor, sigma, dim=-1):
    """
    Applies 1D Gaussian smoothing along a specified dimension of a 3D tensor.

    Args:
        input_tensor (torch.Tensor): A 3D tensor.
        sigma (float): Standard deviation for the Gaussian kernel.
        dim (int): The dimension along which to apply smoothing.
                   Default is -1 (last dimension).

    Returns:
        torch.Tensor: The smoothed tensor with the same shape as input_tensor.
    """

    # Convert NumPy array to PyTorch tensor
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor).float()
        is_numpy = True

    # Create the Gaussian kernel and move it to the same device as the input
    kernel = make_gaussian_kernel(sigma).to(input_tensor.device)
    # Reshape kernel to (1, 1, kernel_size) for conv1d
    kernel = kernel.view(1, 1, -1)
    padding = kernel.shape[-1] // 2

    # Permute the tensor so that the smoothing dimension becomes the last dimension.
    dims = list(range(input_tensor.dim()))
    # Remove the chosen dimension and append it to the end.
    dims.remove(dim)
    dims.append(dim)
    input_perm = input_tensor.permute(*dims)

    # Flatten all dimensions except the last one into a single batch dimension.
    orig_shape = input_perm.shape  # e.g. (D0, D1, L) where L is the smoothing dimension
    flattened = input_perm.contiguous().view(-1, orig_shape[-1])
    # Add a dummy channel dimension to match conv1d input (N, C, L)
    flattened = flattened.unsqueeze(1)

    # Apply 1D convolution (Gaussian smoothing)
    smoothed_flat = F.conv1d(flattened, kernel, padding=padding)
    # Remove the dummy channel dimension
    smoothed_flat = smoothed_flat.squeeze(1)

    # Reshape back to the permuted tensor shape (with possibly same size along the smoothed dimension)
    new_shape = list(orig_shape)
    new_shape[-1] = smoothed_flat.shape[-1]
    smoothed_perm = smoothed_flat.view(*new_shape)

    # Compute the inverse permutation to return to the original tensor order.
    inverse = [0] * len(dims)
    for i, d in enumerate(dims):
        inverse[d] = i
    smoothed = smoothed_perm.permute(*inverse)
    
    if is_numpy:
        smoothed = smoothed.numpy()

    return smoothed

def gaussian_smoothing_3d(volume, sigma, device = None):
    """
    Apply 3D Gaussian filter using F.conv3d with separable convolutions.
    
    Args:
        volume: 3D numpy array
        sigma: Sigma for Gaussian kernel
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        Filtered volume as numpy array
    """
    
    # Determine device
    if device is None:
        device = io.get_available_devices()

    # Convert numpy array to PyTorch tensor with explicit data type
    x = torch.from_numpy(volume.astype(np.float32)).to(device=device, dtype=torch.float32)
    
    # Calculate kernel size (needs to be odd)
    kernel_size = int(2 * 3 * sigma + 1)  # 3-sigma rule
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    # Create 1D Gaussian kernel with explicit float32 dtype
    kernel_1d = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, 
                         dtype=torch.float32, device=device)**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Add batch and channel dimensions for processing
    x = x.unsqueeze(0).unsqueeze(0)
    
    # Apply the convolutions sequentially using F.conv3d
    with torch.no_grad():  # No need to track gradients
        # Z dimension
        kernel = kernel_1d.view(1, 1, 1, 1, kernel_size).to(dtype=torch.float32)
        padding_z = (0, 0, kernel_size // 2)
        y = F.conv3d(x.to(dtype=torch.float32), 
                   kernel, 
                   padding=padding_z)
        
        # Y dimension
        kernel = kernel_1d.view(1, 1, 1, kernel_size, 1).to(dtype=torch.float32)
        padding_y = (0, kernel_size // 2, 0)
        y = F.conv3d(y.to(dtype=torch.float32), 
                   kernel, 
                   padding=padding_y)
        
        # X dimension
        kernel = kernel_1d.view(1, 1, kernel_size, 1, 1).to(dtype=torch.float32)
        padding_x = (kernel_size // 2, 0, 0)
        y = F.conv3d(y.to(dtype=torch.float32), 
                   kernel, 
                   padding=padding_x)
    
    # Convert back to numpy with explicit conversion to float32 first
    if device.type == "cpu":
        return y.to(dtype=torch.float32).squeeze().numpy()
    else:
        del kernel_1d, kernel, x
        gc.collect()
        torch.cuda.empty_cache()
        return y.to(dtype=torch.float32).squeeze().cpu().numpy()


