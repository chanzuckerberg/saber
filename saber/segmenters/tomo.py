from saber.utils import preprocessing as preprocess
from saber.segmenters.base import saber3Dsegmenter
from saber.filters import masks as mask_filters
import saber.visualization.results as cryoviz
import saber.visualization.sam2 as vidviz
import saber.filters.gaussian as gauss
from tqdm import tqdm
import numpy as np
import torch

class cryoTomoSegmenter(saber3Dsegmenter):
    def __init__(self,
        sam2_cfg: str = 'base', 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025
    ):      
        """
        Initialize the cryoTomoSegmenter
        """ 
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area)

        # Flag to Bound the Segmentation to the Tomogram
        self.filter_segmentation = True
        self.bound_segmentation = True


    def generate_slab(self, vol, zSlice, slab_thickness):
        """
        Generate a Slab of the Tomogram at a Given Depth
        """

        # Project a Single Slab 
        self.image0 = preprocess.project_tomogram(vol, zSlice, slab_thickness)
        self.image0 = preprocess.contrast(self.image0, std_cutoff=3)
        self.image0 = preprocess.normalize(self.image0)
        self.image = np.stack([self.image0, self.image0, self.image0], axis=-1)

        return self.image

    @torch.inference_mode()
    def segment_slab(self, vol, slab_thickness, zSlice=None, display_image=True):
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
        self.generate_slab(self.vol, zSlice, slab_thickness)

        # Segment Slab 
        self.segment_image(self.image, display_image = display_image)

        return self.masks

    def segment(
        self, 
        vol,
        slab_thickness: int,
        zSlice: int = None,
        save_run: str = None, 
        show_segmentations: bool = False, 
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """
        return self.segment_vol(vol, slab_thickness, zSlice, save_run, show_segmentations)

    @torch.inference_mode()
    def segment_vol(
        self, 
        vol,
        slab_thickness: int,
        zSlice: int = None,
        save_run: str = None, 
        show_segmentations: bool = False, 
    ):  
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if not show_segmentations:  save_mask = True
        else:                       save_mask = False
        self.is_tomogram_mode = True        

        # Segment Initial Slab 
        self.segment_slab(vol, slab_thickness, zSlice, display_image=False)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if save_mask and save_run is not None: # TODO: Figure out a better name / method for this.
            cryoviz.save_slab_segmentation(save_run, self.image, self.masks)        
            
        # Check to Make Sure Masks are Found
        if len(self.masks) == 0:
            return None

        # If A Mask is Found, Follow to 3D Segmentation Propagation

        # Initialize Video Predictor
        if self.inference_state is None:
            self.inference_state = self.video_predictor.create_inference_state_from_tomogram(self.vol)  

        # Set up score capture hook
        captured_scores, hook_handle = self._setup_score_capture_hook()                  

        # Get the dimensions of the volume.
        (nx, ny, nz) = (
            len(self.inference_state['images']),
            self.masks[0]['segmentation'].shape[0],
            self.masks[0]['segmentation'].shape[1]
        )

        # Set annotation frame
        self.ann_frame_idx = zSlice if zSlice is not None else nx // 2 

        # Add masks to predictor
        self._add_masks_to_predictor(self.masks, self.ann_frame_idx, ny)

        # Propagate and filter
        mask_shape = (nx, ny, nz)
        vol_masks, video_segments = self._propagate_and_filter(
            self.vol, self.masks, captured_scores, mask_shape,
            filter_segmentation=self.bound_segmentation,
            show_segmentations=show_segmentations
        )

        # Remove hook and Reset Inference State
        hook_handle.remove()
        self.video_predictor.reset_state(self.inference_state)

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
        sam2_cfg: str = 'base', 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025
    ):
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, min_rel_box_size)
    """
    Initialize the multiDepthTomoSegmenter
    """

    def segment(self,
        vol,
        slab_thickness: int,
        num_slabs: int = 3,
        zSlice: int = None,
        save_run: str = None, 
        show_segmentations: bool = False, 
    ):
        """
        Segment a 3D tomogram using the Video Predictor
        """
        if self.target_class > 0 or self.classifier is None:
            return self.single_segment(vol, slab_thickness, num_slabs)
        else:
            print("Multiclass Segmentation is not implemented yet")
            # return self.multiclass_segment(vol, slab_thickness, num_slabs)

    @torch.inference_mode()
    def single_segment(self, vol, slab_thickness, num_slabs):
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
            offset = (i - num_slabs // 2) * slab_thickness
            slab_center = int(center_index + offset)
            
            # Segment this slab
            masks3d = self.segment_vol(vol, slab_thickness, zSlice=slab_center, show_segmentations=False)        

            # Convert to binary
            masks3d = (masks3d > 0).astype(np.uint16)

            # Update final masks with maximum operation (in-place)
            np.maximum(combined_mask, masks3d, out=combined_mask)

        # (TODO): Operation to Separate the Segmentation Masks
        combined_mask = self.separate_masks(combined_mask)

        # Apply Adaptive Gaussian Smoothing to the Segmentation Mask              
        combined_mask = mask_filters.fast_3d_gaussian_smoothing(
            combined_mask, scale=0.075, deviceID=self.deviceID) 

        return combined_mask

    def separate_masks(self, combined_mask):
        """
        Separate the Segmentation Masks
        """
        # TODO: Implement this
        return combined_mask

