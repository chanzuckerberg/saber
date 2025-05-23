from saber.classifier.datasets.RandMaskCrop import crop_and_resize_adaptive
from monai.transforms import NormalizeIntensity
from saber.classifier.models import common
import saber.utilities as utils
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
        self.device = utils.get_available_devices(deviceID)
        
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
        
        # If image is a single image ([1, H, W]) but we have multiple masks, expand it
        if image.shape[0] == 1 and masks.shape[0] > 1:
            image = image.expand(masks.shape[0], -1, -1, -1)
        
        # Binarize the masks (assumes masks shape is [Nmasks, H, W])
        binarized_masks = (masks > 0).to(torch.uint8)
        
        # Compute the area of each mask and filter small ones
        mask_areas = binarized_masks.sum(dim=[1, 2])
        valid_indices = (mask_areas >= self.min_area).nonzero(as_tuple=False).squeeze(1).tolist()
        binarized_masks = binarized_masks[valid_indices]
        if binarized_masks.shape[0] == 0:
            return None, []
        
        # Add a channel dimension to masks → [Nmasks, 1, H, W]
        binarized_masks = binarized_masks.unsqueeze(1)
        
        # Now ensure the image batch matches the number of valid masks
        if image.shape[0] == 1 and binarized_masks.shape[0] > 1:
            im_batch = image.expand(binarized_masks.shape[0], -1, -1, -1)
        else:
            im_batch = image
        
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
        # Normalize the image and preprocess inputs into a batch
        
        # Convert to PyTorch tensors if needed
        image = torch.tensor(image, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.uint8)  # Shape: (Nclass, H, W)
        
        # Apply Transforms and Preprocess Inputs
        image = self.transforms(image)
        image, masks = self.apply_crops(image, masks)
        input_tensor, valid_indices = self.preprocess(image, masks)

        if input_tensor is None:
            return None

        # Perform inference
        with torch.no_grad():
            if self.model.input_mode == 'separate':
                input_batch = input_tensor[:,0,].unsqueeze(1)
                input_masks = input_tensor[:,1,].unsqueeze(1)
                logits = self.model(input_batch,input_masks)     # Forward pass → (Nmasks, num_classes)
            else:
                logits = self.model(input_tensor)     # Forward pass → (Nmasks, num_classes)
            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        probs = probs.cpu().numpy()               # Convert to NumPy for easy saving

        # Assign predicted probabilities to valid mask positions
        full_probs = np.zeros((masks.shape[0], probs.shape[1]), dtype=np.float32)
        if valid_indices:
            full_probs[valid_indices] = probs

        return full_probs  # Shape: (Nmasks, num_classes)
    
    def apply_crops(self, image, masks):
        """
        Applies crops to the input tensor.
        Args:
            data (Tensor): A tensor of shape (N, 2, H, W) where data[i, 0] is the image
                        and data[i, 1] is the mask.
        Returns:
            Tensor: A tensor of cropped images and masks with shape (N, 2, H_out, W_out).
        """
        
        
        # Determine the Number of Images and Iterate 
        nImages = masks.shape[0]
        image0 = image.unsqueeze(0)
        for ii in range(nImages):
            
            # TODO: Reshape Image 
            image, mask = crop_and_resize_adaptive(image0, masks[ii,])
            
            # Initialize Output Tensor if First Iteration
            if ii == 0:
                output = torch.zeros([nImages, 2, image.shape[1], image.shape[2]])
            # Concatenate the Cropped Image and Mask
            output[ii,] = torch.cat([image, mask], dim=0)
        
        return output[:,0], output[:,1]
