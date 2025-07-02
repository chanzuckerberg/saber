import numpy as np
import torch

class FourierRescale3D:
    def __init__(self, input_voxel_size, output_voxel_size):
        """
        Initialize the FourierRescale operation with voxel sizes.

        Parameters:
            input_voxel_size (int or tuple): Physical spacing of the input voxels (d, h, w)
                                             or a single int (which will be applied to all dimensions).
            output_voxel_size (int or tuple): Desired physical spacing of the output voxels (d, h, w)
                                              or a single int (which will be applied to all dimensions).
                                              Must be greater than or equal to input_voxel_size.
        """
        # Convert to tuples if single int is provided.
        if isinstance(input_voxel_size, int) or isinstance(input_voxel_size, float):
            input_voxel_size = (input_voxel_size, input_voxel_size, input_voxel_size)
        if isinstance(output_voxel_size, int) or isinstance(output_voxel_size, float):
            output_voxel_size = (output_voxel_size, output_voxel_size, output_voxel_size)
            
        self.input_voxel_size = input_voxel_size
        self.output_voxel_size = output_voxel_size
        
        # Check: output voxel size must be greater than or equal to input voxel size (element-wise).
        if any(out_vs < in_vs for in_vs, out_vs in zip(input_voxel_size, output_voxel_size)):
            raise ValueError("Output voxel size must be greater than or equal to the input voxel size.")

        # Determine device: use GPU if available, otherwise CPU.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
  
    def run(self, volume):
        """
        Rescale a 3D volume (or a batch of volumes on GPU) using Fourier cropping.
        """
        # Initialize return_numpy flag
        return_numpy = False

        # If a numpy array is passed, convert it to a PyTorch tensor.
        if isinstance(volume, np.ndarray):
            return_numpy = True
            volume = torch.from_numpy(volume)

        # If running on CPU, ensure only a single volume is provided.
        if self.device.type == 'cpu' and volume.dim() == 4:
            raise AssertionError("Batched volumes are not allowed on CPU. Please provide a single volume.")

        if volume.dim() == 4:
            output = self.batched_rescale(volume)
        else: 
            output = self.single_rescale(volume)
        
        # Return to CPU if Compute is on GPU
        if self.device == torch.device('cuda'):
            output = output.cpu()
            torch.cuda.empty_cache()

        # Either return a numpy array or a torch tensor
        if return_numpy:
            return output.numpy()
        else:
            return output

    def batched_rescale(self, volume: torch.Tensor):
        """
        Process a (batched) volume: move to device, perform FFT, crop in Fourier space,
        and compute the inverse FFT.
        """
        volume = volume.to(self.device)
        is_batched = (volume.dim() == 4)
        if not is_batched:
            volume = volume.unsqueeze(0)

        fft_volume = torch.fft.fftn(volume, dim=(-3, -2, -1), norm='ortho')
        fft_volume = torch.fft.fftshift(fft_volume, dim=(-3, -2, -1))
        
        start_d, start_h, start_w, new_depth, new_height, new_width = self.calculate_cropping(volume)
        
        fft_cropped = fft_volume[..., 
                                 start_d:start_d + new_depth,
                                 start_h:start_h + new_height,
                                 start_w:start_w + new_width]
        
        fft_cropped = torch.fft.ifftshift(fft_cropped, dim=(-3, -2, -1))
        out_volume = torch.fft.ifftn(fft_cropped, dim=(-3, -2, -1), norm='ortho')
        out_volume = out_volume.real
        
        if not is_batched:
            out_volume = out_volume.squeeze(0)
            
        return out_volume

    def single_rescale(self, volume: torch.Tensor) -> torch.Tensor:
        return self.batched_rescale(volume)

    def calculate_cropping(self, volume: torch.Tensor):
        """
        Calculate cropping indices and new dimensions based on the voxel sizes.
        """
        in_depth, in_height, in_width = volume.shape[-3:]
        
        # Check if dimensions are odd
        d_is_odd = in_depth % 2
        h_is_odd = in_height % 2
        w_is_odd = in_width % 2
        
        # Calculate new dimensions
        extent_depth = in_depth * self.input_voxel_size[0]
        extent_height = in_height * self.input_voxel_size[1]
        extent_width = in_width * self.input_voxel_size[2]
        
        new_depth = int(round(extent_depth / self.output_voxel_size[0]))
        new_height = int(round(extent_height / self.output_voxel_size[1]))
        new_width = int(round(extent_width / self.output_voxel_size[2]))
        
        # Ensure new dimensions are even
        new_depth = new_depth - (new_depth % 2)
        new_height = new_height - (new_height % 2)
        new_width = new_width - (new_width % 2)
        
        # Calculate starting points with odd/even correction
        start_d = (in_depth - new_depth) // 2 + (d_is_odd)
        start_h = (in_height - new_height) // 2 + (h_is_odd)
        start_w = (in_width - new_width) // 2 + (w_is_odd)
        
        return start_d, start_h, start_w, new_depth, new_height, new_width

class FourierRescale2D:

    @staticmethod
    def run_resolution(image, input_pixsize: float, target_pixsize: float, device: torch.device = None):
        """
        Rescale a 2D image using Fourier cropping.
        """
        scale_factor = target_pixsize / input_pixsize
        if target_pixsize <= input_pixsize:
            raise ValueError(f"Target pixel size ({target_pixsize}Å) must be larger than current pixel size ({input_pixsize}Å)")
        return FourierRescale2D._rescale(image, scale_factor, device)

    @staticmethod
    def run(image, scale_factor: float, device: torch.device = None):
        """
        Rescale a 2D image using Fourier cropping.
        """
        if scale_factor < 1:
            raise ValueError("Scale factor must be greater than 1")
        return FourierRescale2D._rescale(image, scale_factor, device)
    
    @staticmethod
    def _rescale(image, scale_factor: float, device: torch.device = None):
        """
        Internal method that performs the actual rescaling.
        """
        # Set device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check to see if image is a numpy array
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device)
            is_numpy = True
        else:
            image = image.to(device)
            is_numpy = False

        # Get dimensions and check if odd
        h, w = image.shape
        h_is_odd = h % 2
        w_is_odd = w % 2
        
        # Calculate new dimensions
        h_new = int(h / scale_factor)
        w_new = int(w / scale_factor)
        
        # Ensure new dimensions are even
        h_new = h_new - (h_new % 2)
        w_new = w_new - (w_new % 2)
        
        # Compute FFT
        imFFT = torch.fft.fftshift(torch.fft.fft2(image))
        
        # Calculate cropping boundaries with odd/even correction
        h_start = (h - h_new) // 2 + (h_is_odd)
        w_start = (w - w_new) // 2 + (w_is_odd)
        
        # Crop in Fourier space
        imFFT_cropped = imFFT[
            h_start:h_start + h_new,
            w_start:w_start + w_new
        ]
        
        # Inverse FFT
        im_cropped = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(imFFT_cropped)))

        if is_numpy:
            im_cropped = im_cropped.cpu().numpy()
            torch.cuda.empty_cache()  # Clear CUDA cache
        else:
            im_cropped = im_cropped.cpu()

        return im_cropped     

   