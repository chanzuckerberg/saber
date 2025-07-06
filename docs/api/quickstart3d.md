# 3D Tomogram Segmentation Quickstart

This quickstart guide shows you how to use SABER's API to segment 3D tomograms programmatically. You'll learn how to process cryo-ET data, FIB-SEM volumes, or any 3D microscopy data using SAM2's video segmentation capabilities.

## 🎯 What You'll Learn

- Load and preprocess tomogram data
- Initialize 3D segmenters with video propagation
- Apply slab-based segmentation strategies

## 🚀 Basic 3D Segmentation

### Step 1: Load Your Tomogram and Import Modules

Before starting, ensure you have SABER installed and import the necessary modules:

```python
from saber.segmenters.tomo import cryoTomoSegmenter
from saber.classifier.models import common
from saber.visualization import cryosam2 as cryoviz
from copick_utils.io import readers
from saber import io
import numpy as np
import torch
import copick

# Option 1: Load from Copick project
root = copick.from_file("path/to/copick_config.json")
run = root.get_run("your_run_id")
vol = readers.tomogram(run, voxel_size=10, algorithm="denoised")

# Option 2: Load directly from file
# Use MRC-file, or any other data reader.
print(f"Volume shape: {vol.shape}")
```

### Step 2:  Initialize the 3D Segmenter and Classifier
The cryoTomoSegmenter class provides SAM2-based 3D segmentation optimized for tomogram data:

```python
# Create a 3D segmenter with SAM2 video capabilities
segmenter = cryoTomoSegmenter(
    sam2_cfg="large",           # SAM2 model size
    deviceID=0,                 # GPU device ID
    min_mask_area=100,          # Minimum mask area for 3D
    min_rel_box_size=0.025      # Minimum relative box size
)
```

```python
# Optional: If a trained classifier is available
classifier = common.get_predictor(
    model_weights="path/to/model.pth",
    model_config="path/to/config.yaml"
)

# Create segmenter with classifier
segmenter = cryoTomoSegmenter(
    sam2_cfg="large",
    classifier=classifier,
    target_class=1,  # Class ID for your target organelle
    min_mask_area=100
)
```

***Refer to the [Training a Classifier](training.md) page to learn how to train your own domain expert classifier.***

### Step 3: Segment a Single Slab

SABER uses a slab-based approach for tomogram segmentation, which extracts 2D representations from 3D volumes:
```python
# Segment a 2D slab from the tomogram
vol, masks = segmenter.segment_slab(
    vol=vol,
    slab_thickness=10,          # Thickness of the slab in pixels
    zSlice=None,                # Use middle slice if None
    display_image=True          # Show results
)

print(f"Found {len(masks)} segments in the slab")
```

### Step 4: Full 3D Segmentation

For complete 3D segmentation, SABER propagates 2D segmentations through the volume:
```python
# Perform complete 3D segmentation
vol_masks = segmenter.segment(
    vol=vol,
    slab_thickness=10,          # Initial slab thickness
    zSlice=None,                # Starting Z-slice
    show_segmentations=True    # Don't display during processing
)

print(f"3D segmentation shape: {vol_masks.shape}")
```

## Manual 3D Prompting

In cases where we want to generate 3D segmentations from a specific initial 2D prompt, we can use the `saber.general.generalSegmenter` class.

```python

segmenter = generalSegmenter( 'large' )

zSlice = 150
segmenter.segment_slab(vol, 10, zSlice)
masks2D = segmenter.masks

masks3D = segmenter.segment(
    vol,               # Input Volume
    masks2D,           # Initial Segmentation Prompt
    zSlice             # Initial Slice where the segmentation features are present 
)

```

## 🔧 Advanced Configuration

### Resolution Control

As a pre-processing step, we can adjust the tomogram resolution to optimize for GPU memory and processing speed:
```python
# Downsample volume to target resolution
from saber.process.downsample import FourierRescale3D

target_resolution = 20  # Å
current_resolution = 10  # Å

scale = current_resolution / target_resolution
vol = FourierRescale3D.run(vol, scale)
```

### Multi-Slab Context (Experimental)

```python
# Generate multiple slabs for better Z-context
image = segmenter.generate_multi_slab(vol, slab_thickness=32, zSlice=50)
# This creates a 3-channel image from multiple Z-slices
```

## 📚 Next Steps

Now that you've mastered basic 3D segmentation:

- **[2D Quickstart](quickstart2d.md)** - Learn 2D micrograph segmentation
- **[API Overview](overview.md)** - Explore advanced features and customization

