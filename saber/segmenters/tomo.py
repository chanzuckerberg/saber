from saber.segmenters.base import saber3Dsegmenter
import saber.visualization.cryosam2 as cryoviz
import saber.process.gaussian_smooth as gauss
import saber.visualization.sam2 as viz
import saber.utilities as utils
from scipy import ndimage
import numpy as np
import torch

class cryoTomoSegmenter(saber3Dsegmenter):
    def __init__(self,
        sam2_cfg: str, 
        deviceID: int = 0,
        classifier = None,
        target_class: int = 1,
        min_mask_area: int = 100,
        min_rel_box_size: float = 0.025
    ):      
        """
        Initialize the cryoTomoSegmenter
        """ 
        super().__init__(sam2_cfg, deviceID, classifier, target_class, min_mask_area, min_rel_box_size)

        # Flag to indicate we're processing a 3D tomogram rather than a 2D image
        self.is_tomogram_mode = False

        # Flag to Bound the Segmentation to the Tomogram
        self.bound_segmentation = True                

    def generate_slab(self, zSlice, slab_thickness):
        """
        Generate a Slab of the Tomogram at a Given Depth
        """

        # Option 2: Project Single Slab 
        image0 = utils.project_tomogram(self.vol, zSlice, slab_thickness)
        image0 = utils.contrast(image0, std_cutoff=3)
        image0 = utils.normalize(image0)
        image = np.stack([image0, image0, image0], axis=-1)

        return image

    @torch.inference_mode()
    def segment(
        self, 
        vol,
        run,
        slab_thickness: int,
        show_segmentations: bool = False, 
        save_run: str = None, 
        zSlice: int = None
    ):  
        """
        Segment a 3D tomogram using the Video Predictor
        """

        # Testing Try 1D Smoothing along Z-Dimension
        vol = gauss.gaussian_smoothing(vol, 5, dim=0)
        vol = utils.normalize(vol)

        # Determine if We Should Show the 2D Segmentations or Show the Segmentations in 3D
        if not show_segmentations:  save_mask = True
        else:                       save_mask = False
        self.is_tomogram_mode = True

        # Set up a dictionary to capture the object score logits from the mask decoder.
        # The keys will be frame indices and the values will be a list of score arrays from that frame.
        captured_scores = {}

        # We'll use an attribute to store the current frame index. It will be updated in propagate_segementation.
        self.current_frame = None

        # Define a Hook to Capture the Object Score Logits
        def mask_decoder_hook(module, inputs, output):
            """
            This hook captures the object score logits every time the SAM mask decoder is run.
            The expected output tuple is: (low_res_multimasks, ious, sam_output_tokens, object_score_logits)
            Since IoUs aren't provided in your version, we capture the object score logits (element index 3).
            """
            # Convert logits from bfloat16 to float32 before converting to NumPy.
            logits = output[3].detach().cpu().to(torch.float32).numpy()

            frame_idx = self.current_frame
            if frame_idx not in captured_scores:
                captured_scores[frame_idx] = []
            captured_scores[frame_idx].append(logits)

        # Register the hook on the SAM mask decoder.
        hook_handle = self.video_predictor.predictor.sam_mask_decoder.register_forward_hook(mask_decoder_hook)

        # Generate Slab
        image = self.generate_slab(zSlice, slab_thickness)

        # Segment Slab 
        self.segment_image( display_image = save_mask)

        # Optional: Save Save Segmentation to PNG or Plot Segmentation with Matplotlib
        if save_mask:
            cryoviz.save_mask_segmentation(run, image, self.masks, save_run)        
            
        # Check to Make Sure Masks are Found
        if len(self.masks) == 0:
            hook_handle.remove()
            return None

        # Get the dimensions of the volume.
        (nx, ny, nz) = (
            len(self.inference_state['images']),
            self.masks[0]['segmentation'].shape[0],
            self.masks[0]['segmentation'].shape[1]
        )

        # the frame index we interact with
        if zSlice is None:
            self.ann_frame_idx = int( nx // 2) 
        else:
            self.ann_frame_idx = int( zSlice )        

        # Extract centers of mass for each mask (for prompting).
        auto_points = np.array([
            ndimage.center_of_mass(item['segmentation'])
            for item in self.masks ])[:, ::-1]

        # Map Segmentation back to full resolution and positive points for segmentation
        prompts = {}
        scale = self.video_predictor.predictor.image_size / ny
        labels = np.array([1], np.int32)
        for ii in range(auto_points.shape[0]):

            sam_points = ( auto_points[ii,:] * scale ).reshape(1, 2)
            ann_obj_id = ii + 1 # give a unique id to each object we interact with (it can be any integers)

            # Predict with Masks
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                mask=self.masks[ii]["segmentation"],
            )

            prompts.setdefault(ann_obj_id, {})
            prompts[ann_obj_id].setdefault(self.ann_frame_idx, [])
            prompts[ann_obj_id][self.ann_frame_idx].append((sam_points, labels)) 

        # Propagate Segmentation in 3D
        mask_shape = (nx, ny, nz)
        vol_masks, video_segments = self.propagate_segementation( mask_shape )
        hook_handle.remove()

        # Filter out low confidence masks at edges of tomograms
        if self.bound_segmentation:
            self.frame_scores = np.zeros([vol.shape[0], len(self.masks)])
            vol_masks, video_segments = self.filter_video_segments(video_segments, captured_scores, mask_shape)
        else: # Convert Video Segments to Masks (Without Filtering)
            vol_masks = utils.convert_segments_to_mask(video_segments, vol_masks, mask_shape, len(self.masks))

        # (Optional) Display Segmentations
        if show_segmentations:
            viz.display_video_segmentation(video_segments, self.inference_state)

        # Reset Inference State
        self.video_predictor.reset_state(self.inference_state)

        return vol_masks

    def generate_multi_slab(self, vol, slab_thickness, zSlice):
        """
        Highly Experimental, Instead of Generating a Slab at a Single Depth,
        Generate 3 Slabs to Provide Z-Context.
        """
        
        # Option 1: Project Multiple Slabs to Provide Z-Context
        image1 = utils.project_tomogram(vol, zSlice - slab_thickness/3, slab_thickness)
        image2 = utils.project_tomogram(vol, zSlice, slab_thickness)
        image3 = utils.project_tomogram(vol, zSlice + slab_thickness/3, slab_thickness)

        # # Extend From Grayscale to RGB 
        image = np.stack([image1, image2, image3], axis=-1)
        image = utils.contrast(image, std_cutoff=3)
        # Normalize the Image to [0,1]        
        image = utils.normalize(image, rgb = True)

        # Hold Onto Original Image for Training
        self.image = image