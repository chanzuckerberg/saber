import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# Custom paired transform for random horizontal flipping
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

# Custom paired transform for random vertical flipping
class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask

# Custom paired transform for random rotation
class RandomRotationPair:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, resample=Image.BILINEAR)
        # Use NEAREST interpolation for masks to preserve labels
        mask = F.rotate(mask, angle, resample=Image.NEAREST)
        return image, mask

# Custom paired transform for color jittering (applied only on the image)
class ColorJitterPair:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(brightness=brightness,
                                             contrast=contrast,
                                             saturation=saturation,
                                             hue=hue)

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


def get_training_transforms_rgb(output_size=(320, 320)):
    def transform(data):
        # Extract image and mask from the dictionary
        image, mask = data["image"], data["mask"]
        
        # Convert to PIL images if they are not already; 
        # This is useful if your images are NumPy arrays.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)

        # Ensure the image is in RGB (if not already)
        image = image.convert("RGB")
        
        # Resize both image and mask
        image = image.resize(output_size, resample=Image.BILINEAR)
        mask = mask.resize(output_size, resample=Image.NEAREST)
        
        # Apply paired random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # Apply paired random vertical flip
        if random.random() < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # Apply paired random rotation
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle, resample=Image.BILINEAR)
        mask = F.rotate(mask, angle, resample=Image.NEAREST)
        
        # Optionally apply a color jitter only on the image
        # (this transform is applied only to the image, not the mask)
        # For a detailed jitter, consider using torchvision.transforms.ColorJitter.
        # Here's a quick example using brightness adjustment:
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            image = F.adjust_brightness(image, brightness_factor)
        
        # Convert back to tensor if needed (or you can return PIL images and 
        # use MONAI's ToTensor or similar transforms later in the pipeline)
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        
        # If you want to normalize using ImageNet mean and std values,
        # you can do it here. Example:
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        
        # Update dictionary and return
        data["image"] = image
        data["mask"] = mask
        return data

    return transform

# Validation transform function (no random augmentations, only resizing and normalization)
def get_validation_transforms_rgb(output_size=(320, 320)):
    def transform(image, mask):
        image = image.convert("RGB")
        image = image.resize(output_size, resample=Image.BILINEAR)
        mask = mask.resize(output_size, resample=Image.NEAREST)
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        return image, mask
    return transform