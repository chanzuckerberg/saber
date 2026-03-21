# 2D Micrograph Segmentation

This guide walks through segmenting 2D microscopy images with SABER's Python API — from loading data to running SAM2 or SAM3 inference with an optional domain expert classifier.

---

## Step 1: Load Your Data

SABER supports `.mrc`, `.tif`, `.png`, and other common microscopy formats. The `io.read_micrograph` helper returns the image as a NumPy array along with the pixel size from the file metadata.

```python
from saber.segmenters.micro import cryoMicroSegmenter
from saber.adapters.base import SAM2AdapterConfig, SAM3AdapterConfig
from saber.classifier.models import common
from saber.utils import io
import numpy as np

image, pixel_size = io.read_micrograph("path/to/micrograph.mrc")
print(f"Image shape: {image.shape}, Pixel size: {pixel_size} Å")
```

---

## Step 2: Choose a Backend and Initialize the Segmenter

=== "SAM2 (Recommended)"

    SAM2 generates all possible masks in the image, then filters them by class using your trained classifier. If you do not yet have a classifier, SAM2 still works in zero-shot mode — it will return everything it detects.

    !!! info "When to use SAM2"
        SAM2 is the right choice when you want **high recall** of a specific organelle class, plan to train a domain expert classifier, or need **3D propagation** in tomographic data. Because it proposes all segments first and filters second, it is robust to unusual density profiles that might confuse a text prompt.

    **Without a classifier — zero-shot mode:**

    ```python
    segmenter = cryoMicroSegmenter(
        cfg=SAM2AdapterConfig(cfg="large"),  # (1)
        deviceID=0,
        min_mask_area=50,                    # (2)
    )
    masks = segmenter.segment(image, display=True)
    ```

    1. Model size: `tiny` / `small` / `base` / `large`. Larger = better quality, more memory.
    2. Masks smaller than this area (in pixels) are discarded.

    **With a trained classifier:**

    ```python
    predictor = common.get_predictor(
        model_weights="results/best_model.pth",
        model_config="results/model_config.yaml",
    )

    segmenter = cryoMicroSegmenter(
        cfg=SAM2AdapterConfig(cfg="large", classifier=predictor),  # (1)
        min_mask_area=50,
    )
    masks = segmenter.segment(image, target_class=1, display=True)  # (2)
    ```

    1. The classifier is passed inside the config. It also auto-fills the SAM2 model size and AMG parameters from the saved classifier config.
    2. `target_class` selects the class index from your classifier. Use `-1` for semantic segmentation (all classes).

    ??? tip "Training your own classifier"
        See the [Training Guide](training.md) for a step-by-step walkthrough. Once trained, the model config YAML carries all the AMG settings — you don't need to tune them separately.

=== "SAM3 (Text-Prompt)"

    SAM3 takes a plain-language description of the target structure. No classifier training is needed — just describe what you are looking for.

    !!! info "When to use SAM3"
        SAM3 is ideal for **quick exploration** of a new dataset, or when you cannot collect enough annotated examples to train a reliable classifier. It is the fastest way to get masks out of a new dataset. However, for specialized structures with unusual contrast (e.g. carbon contamination vs. organelles), a SAM2 + classifier combination will generally outperform text prompting.

    ```python
    segmenter = cryoMicroSegmenter(cfg=SAM3AdapterConfig())
    masks = segmenter.segment(image, text="ribosome", display=True)
    ```

    !!! warning "Precision on domain-specific data"
        SAM3 is trained on broad biological vocabulary. Highly specialized structures — or structures that only differ from background by subtle density changes — may not be accurately described by text alone. If results are inconsistent, consider annotating a small dataset and training a SAM2 classifier.

---

## Step 3: Review the Output

`segment()` returns a list of mask dictionaries, one per detected instance:

```python
print(f"Found {len(masks)} segments")

# Each mask dict contains:
# masks[i]['segmentation'] — (H, W) boolean array
# masks[i]['area']         — pixel count
# masks[i]['bbox']         — [x, y, w, h]
```

---

## Advanced: Resolution Control

Fourier-crop the image before inference to reduce memory usage and speed up processing. This is recommended when images exceed ~2048 pixels in either dimension.

```python
from saber.filters.downsample import FourierRescale2D

scale = 2  # Downsample by 2x
image = FourierRescale2D.run(image, scale)
```

!!! tip "Target Resolution"
    For cryo-EM data, aim for a pixel size of ~8–15 Å after downsampling. Most organelles are well-resolved at this range and GPU memory stays manageable with any SAM2 model size.

## Advanced: Sliding Window for High-Resolution Images

When you need to preserve full resolution (e.g. for very small particles), use sliding window inference. The image is divided into overlapping patches, each segmented independently, and results are merged.

```python
segmenter = cryoMicroSegmenter(
    cfg=SAM2AdapterConfig(cfg="large"),
    min_mask_area=100,
    window_size=512,    # Patch size in pixels
    overlap_ratio=0.5,  # Overlap between adjacent patches
)

masks = segmenter.segment(
    image0=large_image,
    use_sliding_window=True,
    display=True,
)
```

!!! note "Experimental"
    Sliding window inference can produce duplicate or split masks at patch boundaries. Review results carefully and tune `overlap_ratio` if boundary artifacts appear.

---

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-arrow-right-24: **3D Quickstart**](quickstart3d.md)

    Apply the same workflow to full tomographic volumes with 3D propagation.

-   [:octicons-arrow-right-24: **Train a Classifier**](training.md)

    Improve SAM2 precision by training a domain expert classifier on your annotations.

-   [:octicons-arrow-right-24: **Parallel Inference**](parallel-inference.md)

    Scale up to full datasets with multi-GPU batch processing.

</div>
