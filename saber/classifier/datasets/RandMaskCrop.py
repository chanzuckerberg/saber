from monai.transforms import MapTransform
import torch.nn.functional as F
import torch, random

class AdaptiveCropd(MapTransform):
    """
    A MONAI MapTransform that crops the image and mask around the mask's bounding box
    (enlarged by a margin), optionally applies a random translation (limited to ±max_translation),
    and then resizes to a fixed output size. If the mask covers nearly the entire image,
    the full image is resized.
    """
    def __init__(self, keys, margin=1.5, output_size=(320, 320),
                 full_mask_thresh=0.9, apply_translation=False, max_translation=15, 
                 allow_missing_keys=False):
        """
        Args:
            keys (list): List of keys to apply the transform to (e.g., ["image", "mask"]).
            margin (float): Margin fraction to enlarge the bounding box.
            output_size (tuple): Fixed output dimensions (height, width).
            full_mask_thresh (float): Threshold fraction to consider the mask as full-image.
            apply_translation (bool): Whether to apply random translation to the crop.
            max_translation (int): Maximum pixel translation (in any direction).
            allow_missing_keys (bool): Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.margin = margin
        self.output_size = output_size
        self.full_mask_thresh = full_mask_thresh
        self.apply_translation = apply_translation
        self.max_translation = max_translation

    def __call__(self, data):
        d = dict(data)
        image = d["image"]
        mask = d["mask"]
        cropped_image, cropped_mask = crop_and_resize_adaptive(
            image, mask, self.margin, self.output_size, self.full_mask_thresh,
            self.apply_translation, self.max_translation
        )
        d["image"] = cropped_image
        d["mask"] = cropped_mask
        return d

def crop_and_resize_adaptive(image, mask, margin=1.5, output_size=(320, 320),
                             apply_translation=False, max_translation=15, full_mask_thresh=0.9):
    """
    Crop the image and mask based on the bounding box of the mask,
    enlarge the bounding box by a margin, optionally randomly shift the crop
    by up to ±max_translation pixels, and then resize the cropped region 
    to a fixed output size. If the bounding box covers nearly the entire image, 
    simply resize the full image.

    Args:
        image (Tensor): Input tensor of shape:
            - [1, H, W] for grayscale images
            - [1, H, W, 3] for RGB images
        mask (Tensor): Mask tensor of shape [1, H, W] or [H, W].
        margin (float): Fraction to enlarge the bounding box (e.g., 0.2 for 20% extra).
        output_size (tuple): Final output size (height, width).
        full_mask_thresh (float): Threshold fraction to consider the mask as full-image.
        apply_translation (bool): Whether to apply a random translation to the crop.
        max_translation (int): Maximum absolute pixel shift allowed (±max_translation).
        
    Returns:
        resized_image (Tensor): Cropped (or full) and resized image.
        resized_mask (Tensor): Cropped (or full) and resized mask.
    """

    # Handle grayscale vs RGB
    is_rgb = image.dim() == 4 and image.shape[3] == 3
    # if is_rgb:
    #     output_size = (output_size[1], output_size[0], 3)

    # Get Image Dimensions
    B, H, W = image.shape[:3]

    # Ensure mask has a channel dimension.
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    nonzero_indices = torch.nonzero(mask[0])
    if nonzero_indices.numel() == 0:
        # If the mask is empty, fallback to resizing the full image.
        # resized_image = F.interpolate(image.unsqueeze(0), size=output_size,
        #                               mode='bilinear', align_corners=False).squeeze(0)
        resized_image = resize_image(image, output_size)
        resized_mask = F.interpolate(mask.unsqueeze(0), size=output_size,
                                     mode='nearest').squeeze(0)
        return resized_image, resized_mask
    else:
        # Compute the bounding box of the mask.
        y_min = int(nonzero_indices[:, 0].min().item())
        y_max = int(nonzero_indices[:, 0].max().item())
        x_min = int(nonzero_indices[:, 1].min().item())
        x_max = int(nonzero_indices[:, 1].max().item())
        
        # Compute bounding box dimensions and ensure a minimum size of 1.
        bbox_h = max(1, y_max - y_min)
        bbox_w = max(1, x_max - x_min)
        
        # If the bounding box covers nearly the full image, just resize the full image.
        if (bbox_h / H) >= full_mask_thresh and (bbox_w / W) >= full_mask_thresh:
            # resized_image = F.interpolate(image.unsqueeze(0), size=output_size,
            #                               mode='bilinear', align_corners=False).squeeze(0)
            resized_image = resize_image(image, output_size)
            resized_mask = F.interpolate(mask.unsqueeze(0), size=output_size,
                                         mode='nearest').squeeze(0)
            return resized_image, resized_mask
        
        # Compute enlarged crop dimensions based on the margin.
        crop_h = int(bbox_h * (1 + margin))
        crop_w = int(bbox_w * (1 + margin))
        
        # Compute the center of the bounding box.
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        # Compute the top-left corner of the crop.
        top = center_y - crop_h // 2
        left = center_x - crop_w // 2
        
        # Optionally apply a random translation (shift) limited to ±max_translation.
        if apply_translation:
            # Determine the allowable shift range to keep the crop within the image.
            min_shift_y = -top
            max_shift_y = H - (top + crop_h)
            min_shift_x = -left
            max_shift_x = W - (left + crop_w)
            # Restrict the random shift to ±max_translation.
            allowed_shift_y_min = max(min_shift_y, -max_translation)
            allowed_shift_y_max = min(max_shift_y, max_translation)
            allowed_shift_x_min = max(min_shift_x, -max_translation)
            allowed_shift_x_max = min(max_shift_x, max_translation)
            # If the allowed range is empty, default to zero shift.
            if allowed_shift_y_min > allowed_shift_y_max:
                shift_y = 0
            else:
                shift_y = random.randint(allowed_shift_y_min, allowed_shift_y_max)
            if allowed_shift_x_min > allowed_shift_x_max:
                shift_x = 0
            else:
                shift_x = random.randint(allowed_shift_x_min, allowed_shift_x_max)
            top += shift_y
            left += shift_x
        
        # Clamp the crop to the image boundaries.
        top = max(0, min(top, H - crop_h))
        left = max(0, min(left, W - crop_w))
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        
        # cropped_image = image[:, top:top+crop_h, left:left+crop_w]
        # cropped_mask = mask[:, top:top+crop_h, left:left+crop_w]
        
        # Crop the image and mask based on format
        if is_rgb:
            cropped_image = image[:, top:top+crop_h, left:left+crop_w, :]
        else:
            cropped_image = image[:, top:top+crop_h, left:left+crop_w]
            
        cropped_mask = mask[:, top:top+crop_h, left:left+crop_w]
        
        # Resize the cropped region to the fixed output size
        resized_image = cropped_image
        resized_image = resize_image(cropped_image, output_size)

        resized_mask = F.interpolate(cropped_mask.unsqueeze(0), size=output_size,
                                     mode='nearest').squeeze(0)

        return resized_image, resized_mask

def resize_image(image, output_size):
    """
    Resize an image to the specified output size, handling both grayscale and RGB formats.
    
    Args:
        image (Tensor): Input tensor of shape:
            - [1, H, W] for grayscale images
            - [1, H, W, 3] for RGB images
        output_size (tuple): Target size as (height, width)
        
    Returns:
        Tensor: Resized image with same format as input
    """
    # Check if image is RGB (has a channel dimension at the end)
    is_rgb = image.dim() == 4 and image.shape[3] == 3
    B = image.shape[0]
    
    # For RGB: handle each channel separately
    if is_rgb:
        resized_image = torch.zeros((B, output_size[0], output_size[1], 3), 
                                  dtype=image.dtype, device=image.device)
        for c in range(3):
            channel = image[:, :, :, c]
            resized_channel = F.interpolate(channel.unsqueeze(1), size=output_size,
                                          mode='bilinear', align_corners=False).squeeze(1)
            resized_image[:, :, :, c] = resized_channel
    # For grayscale
    else:
        resized_image = F.interpolate(image.unsqueeze(0), size=output_size,
                                    mode='bilinear', align_corners=False).squeeze(0)
    
    return resized_image
