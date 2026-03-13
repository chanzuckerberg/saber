from saber.utils import preprocessing as preprocess
from saber.segmenters.base import saber3D
from saber.filters import masks as mask_filters
import saber.visualization.results as cryoviz
import saber.filters.gaussian as gauss
from saber.segmenters import utils
from saber.adapters.sam2.amg import cfgAMG
from saber.adapters.base import AdapterConfig
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch

class cryoTomoSegmenter(saber3D):
    def __init__(self,
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        cfg: Optional[AdapterConfig] = None,
        amg_cfg: Optional[cfgAMG] = None,
    ):
        """
        Initialize the cryoTomoSegmenter
        """
        super().__init__(
            cfg, deviceID, classifier, target_class, 
            amg_cfg
        )

        # Threshold for Certainty Aware Distillation
        self.filter_threshold = 0.5

    @torch.inference_mode()
    def segment_slab(self, 
        vol, 
        slab_thickness: int = 10, 
        zSlice: Optional[int] = None, 
        display: bool = True,
        text: Optional[str] = None ):
        """
        Segment a 2D image using the Video Predictor
        """

        # 1D Smoothing along Z-Dimension
        self.vol = gauss.gaussian_smoothing(vol, 5, dim=0)
        self.vol = preprocess.normalize(self.vol)

        # If No Z-Slice is Provided, Use the Middle of the Tomogram
        if zSlice is None:
            zSlice = int(self.vol.shape[0] // 2)
            
        # Generate Slab
        self.image = preprocess.project_tomogram(self.vol, zSlice, slab_thickness)

        # Segment Slab 
        self.segment_image(self.image, display = display, text_prompt=text)

        return self.masks

    def segment(
        self, 
        vol,
        slab_thickness: int,
        zSlice: int = None,
        save_run: str = None, 
        display: bool = False, 
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """
        return self.segment_vol(vol, slab_thickness, zSlice, save_run, display)

    @torch.inference_mode()
    def segment_vol(
        self, 
        vol,
        slab_thickness: int,
        zSlice: int = None,
        save_run: str = None, 
        display: bool = False, 
    ):  
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if display:  save_mask = False
        else:                   save_mask = True
        self.is_tomogram_mode = True        

        # Segment Initial Slab 
        self.segment_slab(vol, slab_thickness, zSlice, display=False)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if save_mask and save_run is not None:
            cryoviz.save_slab_seg(save_run, self.image, self.masks)        
            
        # Check to Make Sure Masks are Found
        if len(self.masks) == 0:
            return None

        # If A Mask is Found, Follow to 3D Segmentation Propagation
        # Initialize Video Predictor via adapter set_volume()
        if not self._vol_loaded:
            self.video_predictor.set_volume(self.vol)
            self._vol_loaded = True

        # Get the dimensions of the volume.
        nx = self.vol.shape[0]
        ny, nz = (
            self.masks[0]['segmentation'].shape[0],
            self.masks[0]['segmentation'].shape[1]
        )

        # Set annotation frame
        self.ann_frame_idx = zSlice if zSlice is not None else nx // 2

        # Propagate and filter (adapter handles seeding + hook + scoring internally)
        mask_shape = (nx, ny, nz)
        vol_masks = self.propagate(mask_shape)
        
        # Display if requested
        if display:
            cryoviz.view_3d_seg(self.vol, vol_masks)
            
        return vol_masks

    def generate_multi_slab(self, vol, slab_thickness, zSlice):
        """
        Highly Experimental, Instead of Generating a Slab at a Single Depth,
        Generate 3 Slabs to Provide Z-Context.
        """
        
        # Option 1: Project Multiple Slabs to Provide Z-Context
        image1 = preprocess.project_tomogram(vol, zSlice - slab_thickness/3, slab_thickness)
        image2 = preprocess.project_tomogram(vol, zSlice, slab_thickness)
        image3 = preprocess.project_tomogram(vol, zSlice + slab_thickness/3, slab_thickness)

        # # Extend From Grayscale to RGB 
        image = np.stack([image1, image2, image3], axis=-1)
        image = preprocess.contrast(image, std_cutoff=3)
        # Normalize the Image to [0,1]        
        image = preprocess.normalize(image, rgb = True)

        # Hold Onto Original Image for Training
        self.image = image


class multiDepthTomoSegmenter(cryoTomoSegmenter):
    def __init__(self,
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        cfg: cfgAMG = None,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025,
        adapter_cfg: Optional[AdapterConfig] = None,
    ):
        """
        Initialize the multiDepthTomoSegmenter
        """
        super().__init__(deviceID, classifier, target_class, cfg, min_mask_area,
                         min_rel_box_size, adapter_cfg=adapter_cfg)

        if target_class < 1: 
            print('[Error]: Multi-Depth Tomogram Segmenter only supports Single-Class Segmentation currently.')
            exit()

    def segment(self,
        vol,
        slab_thickness: int,
        num_slabs: int = 3,
        delta_z: int = 30,
        save_run: str = None, 
        display: bool = False, 
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Store Whether to Show Segmentations
        self.show_segments = display
        
        # Determine Segmentation Mode
        if self.target_class > 0 or self.classifier is None:
            return self.single_segment(vol, slab_thickness, num_slabs, delta_z)
        else:
            print("Multiclass Segmentation is not implemented yet")
            # return self.multiclass_segment(vol, slab_thickness, num_slabs)

    @torch.inference_mode()
    def single_segment(self, vol, slab_thickness, num_slabs, delta_z):
        """
        Segment a 3D tomogram using the Video Predictor
        """
        
        depth = vol.shape[0]
        center_index = depth // 2
        
        # Initialize combined mask with zeros (using volume shape)
        combined_mask = np.zeros((vol.shape), dtype=np.uint16)

        # Process each slab
        for i in tqdm(range(num_slabs)):
            # Define the center of the slab
            offset = (i - num_slabs // 2) * delta_z
            slab_center = int(center_index + offset)

            # --- Bounds check for slab_center ---
            if slab_center < 0 or slab_center >= depth:
                print(f"Skipping slab {i}: slab_center={slab_center} out of range (0–{depth-1})")
                continue            
            
            # Segment this slab
            masks3d = self.segment_vol(
                vol, slab_thickness, 
                zSlice=slab_center, 
                show_segmentations=False
            )        
            if masks3d is None: # Skip if No Masks Found
                continue

            # Convert to binary
            masks3d = (masks3d > 0).astype(np.uint16)

            # Update final masks with maximum operation (in-place)
            np.maximum(combined_mask, masks3d, out=combined_mask)

        # # Apply Adaptive Gaussian Smoothing to the Segmentation Mask              
        # combined_mask = mask_filters.fast_3d_gaussian_smoothing(
        #     combined_mask, scale=0.025, deviceID=self.deviceID) 

        # Operation to Separate the Segmentation Masks
        combined_mask = utils.separate_masks(combined_mask)

        # Display the Segmentation if Requested
        if self.show_segments:
            cryoviz.view_3d_seg(vol, combined_mask)

        return combined_mask