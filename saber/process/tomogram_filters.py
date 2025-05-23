from torch.fft import fftshift, ifftshift, fftn, ifftn
import matplotlib.pyplot as plt
import numpy as np
import torch, math

"""
This module contains functions for creating cosine-low pass filter and applying it to tomograms.
This is a written translation of the MATLAB code cosine_filter.m from the artia-wrapper package
(https://github.com/uermel/artia-wrapper/tree/master)
"""

class Filter3D:
    def __init__(self, apix, sz, lp=0, lpd=0, hp=0, hpd=0, device=None):
        """
        Initialize the Filter3D class.

        Args:
            apix (float): Pixel size in angstrom.
            sz (tuple): Size of the tomogram (D, H, W).
            lp (float): Low-pass cutoff resolution in angstroms.
            lpd (float): Low-pass decay width in pixels.
            hp (float): High-pass cutoff resolution in angstroms.
            hpd (float): High-pass decay width in pixels.
            device (torch.device, optional): Device for the filter tensor.
        """
        # Set Parameters
        self.apix = apix
        self.sz = sz
        self.lp = lp
        self.lpd = lpd
        self.hp = hp
        self.hpd = hpd
        self.dtype = torch.float32

        # Set Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Check if low-pass cutoff resolution is less than high-pass cutoff resolution
        if self.lp > self.hp and self.lp > 0 and self.hp > 0:
            raise ValueError("Low-pass cutoff resolution must be less than high-pass cutoff resolution.")

        # Convert cutoff values from angstroms to pixels
        self.lp_pix = self.angst_to_pix(self.lp) if self.lp > 0 else 0  # Low-pass cutoff in pixels
        self.hp_pix = self.angst_to_pix(self.hp) if self.hp > 0 else 0  # High-pass cutoff in pixels
        # LPD and HPD are always in pixels; do not convert
        self.lpd_pix = self.lpd  # Decay width in pixels
        self.hpd_pix = self.hpd  # Decay width in pixels

        print('Constructing 3D Cosine Filter...')
        self.cosine_filter()

    def angst_to_pix(self, ang):
        """
        Convert angstroms to pixels based on the pixel size.

        Args:
            ang (float): Measurement in angstroms.

        Returns:
            float: Measurement in pixels.
        """
        return max(self.sz) / (ang / self.apix)

    def cosine_filter(self):
        """
        Creates a combined low-pass and high-pass cosine filter for 3D tomograms.
        """
        D, H, W = self.sz

        # Create spatial frequency grids in pixel space (centered at zero)
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=self.device, dtype=self.dtype) - D // 2,
            torch.arange(H, device=self.device, dtype=self.dtype) - H // 2,
            torch.arange(W, device=self.device, dtype=self.dtype) - W // 2,
            indexing='ij'
        )
        r = torch.sqrt(xx**2 + yy**2 + zz**2)  # Radial distance in pixels

        # Low-pass filter
        lpv = self.construct_filter(r, self.lp_pix, self.lpd_pix, mode='lp')

        # High-pass filter
        hpv = self.construct_filter(r, self.hp_pix, self.hpd_pix, mode='hp')

        # Combined Filter
        self.filter = lpv * hpv

    def construct_filter(self, r, freq, freqdecay, mode='lp'):
        """
        Constructs a low-pass or high-pass filter based on the mode.

        Args:
            r (torch.Tensor): Radial spatial frequency tensor in pixels.
            freq (float): Cutoff frequency in pixels.
            freqdecay (float): Decay width in pixels.
            mode (str): 'lp' for low-pass, 'hp' for high-pass.

        Returns:
            torch.Tensor: Filter mask.
        """
        if mode not in ['lp', 'hp']:
            raise ValueError("Mode must be 'lp' for low-pass or 'hp' for high-pass.")

        # Skip filter
        if freq == 0 and freqdecay == 0:
            filter_mask = torch.ones_like(r)
        # Box Filter
        elif freq > 0 and freqdecay == 0:
            filter_mask = (r < freq).float()
            if mode == 'hp': filter_mask = 1 - filter_mask
        # # Cosine Filter starting with 1 (low-pass) or 0 (high-pass)
        # elif freq == 0 and freqdecay > 0:
        #     filter_mask = (r < freq).float()
        #     sel = r <= freqdecay
        #     filter_mask[sel] = 0.5 + 0.5 * torch.cos(math.pi * r[sel] / freqdecay)
        #     if mode == 'lp': filter_mask = 1 - filter_mask
        # Box Filter with cosine decay
        else:
            half_decay = freqdecay / 2.0
            filter_mask = (r < freq).float()
            sel = (r > (freq - half_decay)) & (r < (freq + half_decay))
            filter_mask[sel] = 0.5 + 0.5 * torch.cos(math.pi * (r[sel] - (freq - half_decay)) / freqdecay)
            if mode == 'hp': filter_mask = 1 - filter_mask

        return filter_mask

    def extract_1d_profile(self, axis='x'):
        """
        Extracts a 1D profile from the 3D filter along the specified axis.

        Returns:
            freqs (np.ndarray): Frequency values in cycles per angstrom (1/Å).
            profile (np.ndarray): Filter magnitude values along the specified axis.
        """

        filter_tensor = self.filter.cpu().numpy()
        D, H, W = filter_tensor.shape

        # Determine the axis
        if axis == 'x':
            central_slice = filter_tensor[D // 2, H // 2, :]
            freqs = np.fft.fftfreq(W, d=self.apix)  # 1/Å
        elif axis == 'y':
            central_slice = filter_tensor[D // 2, :, W // 2]
            freqs = np.fft.fftfreq(H, d=self.apix)  # 1/Å
        elif axis == 'z':
            central_slice = filter_tensor[:, H // 2, W // 2]
            freqs = np.fft.fftfreq(D, d=self.apix)  # 1/Å
        else:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        # Only keep positive frequencies
        mask = freqs >= 0
        freqs_positive = freqs[mask]
        profile_positive = central_slice[mask]

        return freqs_positive[::-1], profile_positive

    def apply(self, data):
        """
        Applies the filter to a tomogram.

        Args:
            data (torch.Tensor): Input data tensor of shape (D, H, W).

        Returns:
            torch.Tensor: Filtered data tensor.
        """

        # Assuming 'vol' is your input data
        if isinstance(data, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            data = torch.from_numpy(data).float()  # Ensure the dtype matches your filter

        # Ensure data is on the same device and dtype
        data = data.to(self.device).type(self.dtype)

        # Compute the Fourier Transform of the data
        filtered_data = ifftn(ifftshift(fftshift(fftn(data)) * self.filter)).real

        return filtered_data

    def show_filter(self):
        """
        Displays the 3D filter as a 3D plot.
        """

        # Extract 1D profile
        freqs, profile = self.extract_1d_profile(axis='x')

        # Create a 2x1 plot
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        # Plot the 1D frequency profile
        axs[0].plot(freqs, profile, label='Axis Profile')
        axs[0].set_xlabel('Frequency (1/Å)')
        axs[0].set_ylabel('Filter Magnitude')
        axs[0].set_title('1D Frequency Profile Along Axis')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_xlim(0, max(freqs))  # Set x-axis range

        # Plot the 2D slice of the filter
        (Nx, Ny, Nz) = self.filter.shape
        axs[1].imshow(self.filter[int(Nx//2), :, :], cmap='gray')
        axs[1].axis('off')

        # Show the plot
        # plt.tight_layout()
        plt.savefig('filter.png')

