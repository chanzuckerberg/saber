from saber.classifier.datasets.RandMaskCrop import crop_and_resize_adaptive
from monai.transforms import NormalizeIntensity
from saber.classifier.models import common
from saber.utils import io
import numpy as np
import torch, yaml


class Predictor:
    """
    Predictor class for running inference using the ConvNeXt-based classifier.
    This class loads a trained model, processes input images and masks,
    and returns classification probabilities.
    """
    def __init__(self, 
        model_config: str, 
        model_weights: str,
        min_area: int = 250,
        deviceID: int = 0):
        """
        Initialize the Predictor with a pre-trained ConvNeXt model.

        Args:
            model_weights (str): Path to the model's weight file (.pth).
            num_classes (int, optional): Number of output classes. Default is 2 (binary classification).
            device (str, optional): Device for inference ('cpu' or 'cuda'). Default is 'cpu'.
        """

        # Initialize Attributes
        self.min_area = min_area

        # Load Model Config
        with open(model_config, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = io.get_available_devices(deviceID)
        
        # Load the model architecture with the specified number of classes
        self.model = common.get_classifier_model(
            self.config['model']['backbone'], 
            self.config['model']['num_classes'], 
            self.config['model']['model_size'],
            deviceID=deviceID   )

        # Load the model weights
        self.model.load_state_dict(torch.load(model_weights, weights_only=True))    
        self.model.to(self.device)
        self.model.eval()

        self.transforms = NormalizeIntensity()

    def preprocess(self, image, masks):
        """
        Converts image and masks into a batched 2-channel tensor while filtering out small masks.
        Accepts image as either a single [H, W] image or a batch [B, H, W] of grayscale images.
        
        Args:
            image (torch.Tensor): Grayscale image(s). Either [H, W] or [B, H, W].
            masks (torch.Tensor): Candidate masks with shape (Nmasks, H, W).
        Returns:
            torch.Tensor: Batched input tensor of shape (Nmasks, 2, H, W).
            list: Valid mask indices.
        """
        # If image is [H, W], add a channel dim → [1, H, W]
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # If image is [B, H, W] assume it's batched and add a channel dim if needed
        elif image.ndim == 3:
            # If the number of images equals the number of masks, assume it's a batched input
            if image.shape[0] == masks.shape[0]:
                image = image.unsqueeze(1)  # now [B, 1, H, W]
            # Otherwise assume image is already [C, H, W] (e.g. C=1) and not batched
        
        # Binarize the masks (assumes masks shape is [Nmasks, H, W])
        binarized_masks = (masks > 0).to(torch.uint8)
        
        # Compute the area of each mask and filter small ones
        mask_areas = binarized_masks.sum(dim=[1, 2])
        valid_indices = (mask_areas >= self.min_area).nonzero(as_tuple=False).squeeze(1).tolist()
        
        # Filter masks and corresponding images
        binarized_masks = binarized_masks[valid_indices]
        if binarized_masks.shape[0] == 0:
            return None, []
        
        # Add a channel dimension to masks → [Nmasks, 1, H, W]
        binarized_masks = binarized_masks.unsqueeze(1)
        
        # Handle image batching based on the FILTERED mask count
        if image.shape[0] == 1:
            # Single image - expand to match filtered mask count
            im_batch = image.expand(binarized_masks.shape[0], -1, -1, -1)
        else:
            # Multiple images - select only the valid ones
            im_batch = image[valid_indices]
        
        # Build the input batch according to the model input mode
        if self.model.input_mode == 'separate':
            input_batch = torch.cat([im_batch, binarized_masks], dim=1).to(self.device)
        else:
            roi = im_batch * binarized_masks       # region of interest
            roni = im_batch * (1 - binarized_masks)  # region of non-interest
            input_batch = torch.cat([roi, roni], dim=1).to(self.device)
        
        return input_batch, valid_indices

    def predict(self, image, masks):
        """
        Runs inference on a batch of masks.

        Args:
            image (numpy.ndarray or torch.Tensor): The input image (H, W).
            masks (numpy.ndarray or torch.Tensor): A batch of candidate masks (Nmasks, H, W).

        Returns:
            numpy.ndarray: The predicted class probabilities of shape (Nmasks, num_classes).
        """
        # Convert to PyTorch tensors if needed
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, dtype=torch.uint8)
        
        # Store original mask count
        original_mask_count = masks.shape[0]
        
        # Apply Transforms and Preprocess Inputs
        image = self.transforms(image)
        
        # Apply crops - this returns one cropped image per mask
        cropped_images, cropped_masks = self.apply_crops(image, masks)
        
        # Now preprocess with already cropped images and masks
        input_tensor, valid_indices = self.preprocess(cropped_images, cropped_masks)

        if input_tensor is None:
            return np.zeros((original_mask_count, self.config['model']['num_classes']), dtype=np.float32)

        # Perform inference
        with torch.no_grad():
            if self.model.input_mode == 'separate':
                input_batch = input_tensor[:,0,].unsqueeze(1)
                input_masks = input_tensor[:,1,].unsqueeze(1)
                logits = self.model(input_batch, input_masks)
            else:
                logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        probs = probs.cpu().numpy()

        # Assign predicted probabilities to valid mask positions
        full_probs = np.zeros((original_mask_count, probs.shape[1]), dtype=np.float32)
        if valid_indices:
            full_probs[valid_indices] = probs

        return full_probs        

    def batch_predict(self, image, masks, batch_size=32):
        """
        Runs inference on masks in batches to avoid CUDA memory issues.
        
        Args:
            image (numpy.ndarray or torch.Tensor): The input image (H, W).
            masks (numpy.ndarray or torch.Tensor): A batch of candidate masks (Nmasks, H, W).
            batch_size (int): Maximum number of masks to process at once. Default is 32.
        
        Returns:
            numpy.ndarray: The predicted class probabilities of shape (Nmasks, num_classes).
        """
        # Convert to tensors if needed
        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, dtype=torch.uint8)
        
        total_masks = masks.shape[0]
        num_classes = self.config['model']['num_classes']
        
        # Initialize output array
        all_probs = np.zeros((total_masks, num_classes), dtype=np.float32)
        
        # Process masks in batches
        for start_idx in range(0, total_masks, batch_size):
            end_idx = min(start_idx + batch_size, total_masks)
            
            # Extract current batch of masks
            mask_batch = masks[start_idx:end_idx]
            
            # Run prediction on this batch
            batch_probs = self.predict(image, mask_batch)
            
            # Handle case where all masks in batch were filtered out
            if batch_probs is not None:
                # Store results in the correct position
                all_probs[start_idx:end_idx] = batch_probs
        
        return all_probs        

    def apply_crops(self, image, masks):
        """
        Applies crops to the input tensor.
        """
        nImages = masks.shape[0]
        image0 = image.unsqueeze(0) if image.ndim == 2 else image
        
        # Process first image to get output dimensions
        img_crop, mask_crop = crop_and_resize_adaptive(image0, masks[0])
        
        # Pre-allocate output tensor on the same device as input
        output = torch.zeros([nImages, 2, img_crop.shape[1], img_crop.shape[2]], 
                            device=masks.device, dtype=image.dtype)
        
        # Store first result
        output[0] = torch.cat([img_crop, mask_crop], dim=0)
        
        # Process remaining images
        for ii in range(1, nImages):
            img_crop, mask_crop = crop_and_resize_adaptive(image0, masks[ii])
            output[ii] = torch.cat([img_crop, mask_crop], dim=0)
            
            # Optional: Empty the cache periodically
            if ii % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return output[:,0], output[:,1]