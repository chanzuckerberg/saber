# 3D Tomogram Segmentation

This guide walks through segmenting 3D tomographic volumes with SABER. The core idea is slab-based: a 2D projection of the volume is segmented first, then SAM2's video predictor propagates those masks through the full Z-stack.

---

## Step 1: Load Your Tomogram

SABER uses [Copick](https://copick.github.io/copick/) to access tomographic volumes. This keeps your data organized and makes batch processing straightforward.

```python
from saber.segmenters.tomo import tomoSegmenter
from saber.adapters.base import SAM2AdapterConfig
from saber.classifier.models import common
from copick_utils.io import readers
import copick

root = copick.from_file("path/to/copick_config.json")
run = root.get_run("your_run_id")
vol = readers.tomogram(run, voxel_size=10, algorithm="denoised")

print(f"Volume shape: {vol.shape}")  # (Z, Y, X)
```

!!! tip "Which algorithm to use?"
    Use `"denoised"` whenever available — denoised reconstructions dramatically improve SAM2's ability to detect organelle boundaries, especially for small or low-contrast structures. Fall back to `"wbp"` if denoised is not available in your project.

---

## Step 2: Choose a Backend and Initialize the Segmenter

=== "SAM2 (Recommended)"

    SAM2 is the primary backend for 3D segmentation. Its video predictor was designed for temporal propagation — SABER repurposes this to propagate 2D masks through Z-slices of a tomogram.

    !!! info "Why SAM2 for 3D?"
        SAM2's video memory bank tracks objects across frames, making it naturally suited to the problem of following a structure across the depth axis of a tomogram. This means a single 2D initialization at the central slice is often enough to produce a complete 3D mask.

    **Without a classifier:**

    ```python
    segmenter = tomoSegmenter(
        cfg=SAM2AdapterConfig(cfg="large"),
        deviceID=0,
        min_mask_area=100,
    )
    ```

    **With a trained classifier:**

    ```python
    predictor = common.get_predictor(
        model_weights="results/best_model.pth",
        model_config="results/model_config.yaml",
    )

    segmenter = tomoSegmenter(
        cfg=SAM2AdapterConfig(cfg="large", classifier=predictor),  # (1)
        min_mask_area=100,
    )
    ```

    1. The classifier screens 2D slab masks before propagation, so only confirmed organelle instances are passed to the video predictor. This prevents false positives from propagating through the entire volume.

    ??? tip "Training your own classifier"
        See the [Training Guide](training.md). The saved classifier config automatically supplies the correct SAM2 model size and AMG parameters when loaded.

=== "SAM3 (Text-Prompt)"

    SAM3 can initialize 3D segmentation from a text description. No classifier is needed — the model segments matching structures in the initial slab, and SAM2's video predictor propagates the results.

    !!! info "When to use SAM3 for 3D"
        SAM3 is useful for quick 3D exploration when you don't yet have a classifier. However, text prompts can miss structures that appear in only some Z-depths or that are similar in appearance to background, leading to incomplete 3D masks. For production runs, a SAM2 + classifier combination is more reliable.

    ```python
    from saber.adapters.base import SAM3AdapterConfig

    segmenter = tomoSegmenter(cfg=SAM3AdapterConfig())
    vol_masks = segmenter.segment(vol, thickness=10, text="ribosome")
    ```

    !!! warning "3D propagation is still SAM2-based"
        The text prompt from SAM3 initializes the 2D slab segmentation, but the subsequent 3D propagation still relies on SAM2's video predictor. This means SAM3 is used for the *detection* step, not the propagation step.

---

## Step 3: Segment a Single Slab (Preview)

Before running full 3D segmentation, preview results on a single 2D slab. This is a fast way to verify your settings before committing to a full volume run.

```python
masks = segmenter.segment_slab(
    vol=vol,
    slab_thickness=10,   # (1)
    zSlice=None,         # (2)
    display=True,
)
print(f"Found {len(masks)} segments in the slab")
```

1. Number of Z-slices averaged together to form the slab projection. Thicker slabs reduce noise but blur fine details. 10–20 voxels works well for most data.
2. If `None`, the middle slice of the volume is used. Set to an integer to preview a specific depth.

---

## Step 4: Full 3D Segmentation

Once slab results look good, run the full 3D workflow. SABER segments the initial slab, then propagates masks forward and backward through the Z-stack.

```python
vol_masks = segmenter.segment(
    vol=vol,
    thickness=10,           # Slab thickness for the 2D initialization step
    zSlice=None,            # Starting Z-slice (None = middle of volume)
    target_class=1,         # Class index from your classifier
    display=False,          # Set True to render a 3D view after segmentation
)

print(f"3D segmentation shape: {vol_masks.shape}")  # (Z, Y, X)
```

!!! note "Return value"
    `segment()` returns a `uint16` NumPy array where each unique non-zero value is a distinct instance ID. Zero is background. This can be saved directly to a Copick segmentation or Zarr file.

---

## Manual Start Slice

If the target structure is not visible at the center of the volume, specify a `zSlice` where it is clearly present. The video predictor initializes from that slice and propagates in both directions.

```python
vol_masks = segmenter.segment(
    vol,
    thickness=10,
    zSlice=150,      # Start from a slice where the target is clearly visible
    target_class=1,
)
```

---

## Multi-Depth Segmentation

For structures that are sparse or only partially visible at any single depth, `multiDepthTomoSegmenter` runs segmentation at multiple Z-depths and merges the results. This significantly improves recall for small or irregular organelles.

```python
from saber.segmenters.tomo import multiDepthTomoSegmenter

segmenter = multiDepthTomoSegmenter(
    cfg=SAM2AdapterConfig(cfg="large", classifier=predictor),
    target_class=1,
    min_mask_area=100,
    min_rel_box_size=0.025,  # (1)
)

vol_masks = segmenter.segment(
    vol,
    thickness=10,
    num_slabs=3,    # (2)
    delta_z=30,     # (3)
)
```

1. Minimum bounding box size relative to the image, used to filter very small detections.
2. Number of Z-depths to initialize from. Each produces an independent 3D mask, and results are merged by taking the union.
3. Z-spacing in voxels between each initialization depth.

!!! tip "When to use multi-depth"
    If a single-slab run misses organelles that you know are present in the volume, increase `num_slabs`. The cost is roughly linear — 3 slabs takes ~3x longer than 1.

---

## Advanced: Resolution Control

Fourier-crop the volume to reduce memory usage before segmentation. This is especially useful for large tomograms.

```python
from saber.filters.downsample import FourierRescale3D

target_resolution = 20  # Å
current_resolution = 10  # Å
scale = current_resolution / target_resolution

vol = FourierRescale3D.run(vol, scale)
```

---

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-arrow-right-24: **2D Quickstart**](quickstart2d.md)

    Start with 2D micrograph segmentation before moving to 3D volumes.

-   [:octicons-arrow-right-24: **Train a Classifier**](training.md)

    Improve 3D results by training a classifier on annotated slab projections.

-   [:octicons-arrow-right-24: **Parallel Inference**](parallel-inference.md)

    Process entire Copick projects across multiple GPUs simultaneously.

</div>
