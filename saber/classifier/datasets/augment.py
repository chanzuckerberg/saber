from monai.transforms import (
    Compose, EnsureChannelFirstd, NormalizeIntensityd, Orientationd,
    RandRotate90d, RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, RandGaussianNoised, RandAffined, RandomOrder,
    RandGaussianSmoothd,
)
from saber.classifier.datasets.RandMaskCrop import AdaptiveCropd
from torch.utils.data import random_split

def get_preprocessing_transforms(random_translations=False):
        transforms = Compose([
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"]),
            # Crop around the segmentation with some jitter.
            AdaptiveCropd(
                 keys=["image", "mask"], output_size=(320, 320), 
                 margin=2, apply_translation=random_translations, max_translation=25),
        ])
        return transforms

def get_training_transforms():
    train_transforms = Compose([
        # RandAffined(
        #     keys=["image", "mask"],
        #     prob=0.75,
        #     translate_range=(30, 30),
        #     padding_mode="border",
        #     mode=("bilinear", "nearest")
        # ),
        RandomOrder([
            RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys="image", prob=0.5, factors=(0.85, 1.15)),
            RandShiftIntensityd(keys="image", prob=0.5, offsets=(-0.15, 0.15)),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.85, 1.15)),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=1.5),
            RandGaussianSmoothd(keys="image", prob=0.5, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
        ])
    ])
    return train_transforms

def get_validation_transforms():
    return get_preprocessing_transforms()

def split_dataset(dataset, val_split=0.2):
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])