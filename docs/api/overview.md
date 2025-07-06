# API Overview

## Introduction 

Welcome to the SABER Python API, your programmatic gateway to high-resolution segmentation in electron tomography (cryo-ET) and electron microscopy (EM). Whether you're integrating segmentation into large-scale pipelines or building custom tools, the SABER API gives you full control over the segmentation workflow.

With SABER's modular and extensible API, you can: 

* **Load and process** tomographic datasets and micrographs
* **Interface with SAM2** and custom domain expert classifiers
* **Apply segmentations** in 2D or propagate across 3D volumes
* **Save, visualize, and evaluate** segmentations with ease
* **Build custom workflows** for your specific research needs

The API is designed for flexibility and reproducibility—ideal for researchers, developers, and power users who need programmatic control over their segmentation pipelines.

## 🏗️ Core Architecture

SABER's API is built around three main components:

### 1. **Segmenter Classes** (`saber.segmenters`)
The core segmentation engines that handle SAM2 integration and 3D propagation:

- **`cryoMicroSegmenter`** - 2D micrograph segmentation with SAM2
- **`cryoTomoSegmenter`** - 3D tomogram segmentation with video propagation
- **`saber2Dsegmenter`** - Base class for 2D segmentation workflows
- **`saber3Dsegmenter`** - Base class for 3D segmentation workflows

### 2. **Entry Points** (`saber.entry_points`)
High-level functions that provide complete segmentation workflows:

- **`segment_micrograph_core`** - Complete 2D micrograph processing
- **`segment_tomogram_core`** - Complete 3D tomogram processing
- **Parallel processing** utilities for batch operations

### 3. **Utility Modules**
Supporting functionality for data handling and visualization:

- **`saber.utils.io`** - Data loading and saving utilities
- **`saber.utils.preprocessing`** - Image processing and helper functions
- **`saber.visualization`** - Result visualization and galleries

## 🚀 Quick Start Paths

Choose your entry point based on your needs:

### **2D Micrograph Segmentation**
Perfect for single-particle EM, SEM images, or 2D slices:

* **[2D Quickstart](quickstart2d.md)** - Learn how to segment 2D EM images with SAM2 and SABER

### **3D Tomogram Segmentation** 
Ideal for cryo-ET, FIB-SEM, or any 3D volume data:

* **[3D Quickstart](quickstart3d.md)** - Apply the workflow on full 3D volumes

## 🔧 Key Features

### **Flexible Input Formats**
```python
# Support for various microscopy data formats
from copick_utils.io import readers
from saber.utils import io

# Load micrographs
image, pixel_size = io.read_micrograph("path/to/image.mrc")

# Load tomograms from Copick projects
vol = readers.tomogram(run, voxel_size=10, algorithm="denoised")
```

### **SAM2 Integration**
```python
# Direct access to SAM2 models with custom parameters
from saber.segmenters.micro import cryoMicroSegmenter

segmenter = cryoMicroSegmenter(
    sam2_cfg="large",  # Model size: tiny, base, large
    min_mask_area=50,  # Filter small artifacts
    window_size=256,   # Sliding window for large images
)
```

### **Domain Expert Classifiers**
```python
# Integrate custom classifiers for improved accuracy
from saber.classifier.models import common

classifier = common.get_predictor(
    model_weights="path/to/model.pth",
    model_config="path/to/config.yaml"
)
```

### **3D Propagation**
```python
# Automatic 3D segmentation from 2D annotations
from saber.segmenters.tomo import cryoTomoSegmenter

segmenter = cryoTomoSegmenter(sam2_cfg="large")
vol_masks = segmenter.segment(vol, slab_thickness=32)
```

## 📊 Output Formats

SABER provides flexible output options:

- **Zarr volumes** - Efficient storage for large datasets
- **NumPy arrays** - Direct access for custom analysis
- **Visualization galleries** - PNG galleries for result review
- **Copick integration** - Native support for collaborative annotation

## 📚 Next Steps

Ready to dive deeper? Follow these paths:

* **[2D Quickstart](quickstart2d.md)** - Get started with micrograph segmentation
* **[3D Quickstart](quickstart3d.md)** - Learn 3D tomogram processing

---

*The SABER API is designed to be both powerful and accessible. Start with the quickstarts to see immediate results, then explore the detailed workflows for advanced usage.* 