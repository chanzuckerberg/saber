# 2D Micrograph Segmentation Quickstart

This quickstart guide shows you how to use SABER's API to segment 2D micrographs programmatically. You'll learn the core classes and functions needed to process EM images, SEM data, or any 2D microscopy data.

## 🎯 What You'll Learn

- Load and preprocess micrograph data
- Initialize SAM2 and SAM3-based segmenters
- Apply domain expert classifiers

## 🚀 Basic Segmentation

### Step 1: Load Your Micrograph and Import Modules

Before starting, ensure you have SABER installed and import the necessary modules: SABER supports various file formats commonly used in microscopy:
```python
from saber.segmenters.micro import cryoMicroSegmenter
from saber.adapters.base import SAM2AdapterConfig, SAM3AdapterConfig
from saber.visualization import classifier as viz
from saber.classifier.models import common
from saber.utils import io
import numpy as np
import torch

# Load a micrograph file (supports .mrc, .tif, .png, etc.)
image, pixel_size = io.read_micrograph("path/to/your/micrograph.mrc")
print(f"Image shape: {image.shape}, Pixel size: {pixel_size} Å")
```

### Step 2: Initialize the Segmenter and Classifier

The `cryoMicroSegmenter` class provides SAM2 and SAM3-based segmentation optimized for cryo-EM data. SAM2 supports model sizes ranging from tiny to large; SAM3 uses text prompts with no classifier required. Use `min_mask_area` to filter out small spurious masks.

```python
# SAM2 — automatic mask generation
segmenter = cryoMicroSegmenter(
    adapter_cfg=SAM2AdapterConfig(cfg="large"),  # Model size: tiny, small, base, large
    deviceID=0,                                  # GPU device ID
    min_mask_area=50,                            # Minimum mask area to keep
)
```

```python
# Optional: add a trained classifier to filter false positives
classifier = common.get_predictor(
    model_weights="path/to/model.pth",
    model_config="path/to/config.yaml"
)

segmenter = cryoMicroSegmenter(
    adapter_cfg=SAM2AdapterConfig(cfg="large"),
    classifier=classifier,
    target_class=1,  # Class ID for your target organelle
    min_mask_area=50
)
```

```python
# SAM3 — text-driven segmentation (no classifier needed)
segmenter = cryoMicroSegmenter(adapter_cfg=SAM3AdapterConfig())
```
***Refer to the [Training a Classifier](training.md) page to learn how to train your own domain expert classifier.***

### Step 3: Run Segmentation

Execute the segmentation process with a single function call.

```python
# SAM2 with classifier — override target class per call
masks = segmenter.segment(image0=image, target_class=1, display_image=True)

# SAM3 — text-driven
masks = segmenter.segment(image0=image, text="ribosome", display_image=True)

print(f"Found {len(masks)} segments")
```

## 🔧 Advanced Configuration

### Resolution Control

As a pre-processing step, we can Fourier crop (downsample or images) to a resolution that is suitable for the available GPU. In cases where the memory requirement is too large, either reduce the model size of SAM2 or reduce the image resolution to below 2048. 

```python
# Downsample to target resolution
from saber.process.downsample import FourierRescale2D

scale = 2
image = FourierRescale2D.run(image, scale)
```

### (Experimental) Sliding Window Segmentation

In cases where high-resolution is essential, we can use a sliding window to segment the images. This guarantees we can use the large base SAM2 model and process the full image resolution. We can vary both the window size (in pixels) and overlap ratio for the sliding window.

```python
# Fine-tune SAM2 behavior
segmenter = cryoMicroSegmenter(
    adapter_cfg=SAM2AdapterConfig(cfg="large"),
    min_mask_area=100,          # Larger minimum area
    window_size=512,            # Window Size (Pixels)
    overlap_ratio=0.5           # More overlap
)

# Use sliding window for images larger than 1536x1536
masks = segmenter.segment(
    image0=large_image,
    use_sliding_window=True,    # Enable sliding window
    display_image=True
)
```

## 📚 Next Steps

Now that you've mastered basic 2D segmentation:

- **[3D Quickstart](quickstart3d.md)** - Learn 3D tomogram segmentation
- **[API Overview](overview.md)** - Explore advanced features and customization
