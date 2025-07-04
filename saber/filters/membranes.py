from monai.inferers import SlidingWindowInferer
from saber import pretrained_weights
import torch, logging
import numpy as np

from membrain_seg.segmentation.networks.inference_unet import (
    PreprocessedSemanticSegmentationUnet,
)
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
)
from membrain_seg.segmentation.dataloading.data_utils import (
    store_segmented_tomograms,
)
from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_mirrored_img, get_prediction_transforms
)

def membrain_preprocess(
    data, 
    transforms, 
    device, 
    normalize_data=True
):
    """
    Preprocess tomogram data from numpy array or PyTorch tensor for inference.
    
    Adapted from load_data_for_inference in membrain-seg repository.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Normalize data if requested
    if normalize_data:
        mean_val = np.mean(data)
        std_val = np.std(data)
        data = (data - mean_val) / std_val
    
    # Add channel dimension (C, H, W, D)
    new_data = np.expand_dims(data, 0)
    
    # Apply transforms
    new_data = transforms(new_data)
    
    # Add batch dimension
    new_data = new_data.unsqueeze(0)
    
    # Move to device
    new_data = new_data.to(device)
    
    return new_data


def membrain_segment(
    data,
    sw_batch_size=4,
    sw_window_size=160,
    test_time_augmentation=True,
    normalize_data=True,
    segmentation_threshold=0.0,
    ):
    """
    Segment tomograms using the membrain-seg trained model from in-memory data.
    
    This function is heavily adapted from the segment() function in the membrain-seg
    repository, modified to work with in-memory numpy arrays or PyTorch tensors
    instead of file paths.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        The 3D tomogram data to be segmented.
    sw_window_size: int, optional
        Sliding window size used for inference. Must be a multiple of 32.
    test_time_augmentation: bool, optional
        If True, test-time augmentation is performed.
    normalize_data : bool, optional
        Whether to normalize the input data (default is True).
    segmentation_threshold: float, optional
        Threshold for the membrane segmentation (default: 0.0).

    Returns
    -------
    predictions : torch.Tensor or np.ndarray
        The segmentation predictions (same type as input data).
    """

    # Check input data type for return type matching
    input_is_numpy = isinstance(data, np.ndarray)

    # Load the trained PyTorch Lightning model
    model_checkpoint = pretrained_weights.get_membrain_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    pl_model = PreprocessedSemanticSegmentationUnet.load_from_checkpoint(
        model_checkpoint, map_location=device, strict=False
    )
    pl_model.to(device)
    
    if sw_window_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32!")
    pl_model.target_shape = (sw_window_size, sw_window_size, sw_window_size)

    # Preprocess the data
    transforms = get_prediction_transforms()
    new_data = membrain_preprocess(
        data, transforms, device=torch.device("cpu"), 
        normalize_data=normalize_data
    )
    new_data = new_data.to(torch.float32)

    # Put the model into evaluation mode
    pl_model.eval()

    # Perform sliding window inference on the new data
    roi_size = (sw_window_size, sw_window_size, sw_window_size)
    inferer = SlidingWindowInferer(
        roi_size,
        sw_batch_size, 
        overlap=0.5,
        progress=True,
        mode="gaussian",
        device=torch.device("cpu"),
    )

    # Perform test time augmentation (8-fold mirroring)
    predictions = torch.zeros_like(new_data)
    if test_time_augmentation:
        logging.info("Performing 8-fold test-time augmentation.")
    
    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                mirrored_input = get_mirrored_img(new_data.clone(), m).to(device)
                mirrored_pred = inferer(mirrored_input, pl_model)
                if not (isinstance(mirrored_pred, list) or isinstance(mirrored_pred, tuple)):
                    mirrored_pred = [mirrored_pred]
                correct_pred = get_mirrored_img(mirrored_pred[0], m)
                predictions += correct_pred.detach().cpu()
    
    if test_time_augmentation:
        predictions /= 8.0

    # Remove batch and channel dimensions for output
    predictions = predictions.squeeze(0).squeeze(0)

    # Apply segmentation threshold
    predictions[predictions > segmentation_threshold] = 1
    predictions[predictions <= segmentation_threshold] = 0

    # Return results
    if input_is_numpy:
        return predictions.numpy()
    else:
        return predictions


