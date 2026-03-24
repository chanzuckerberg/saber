# API Overview

Welcome to the SABER Python API. This page covers the core architecture and helps you decide which segmentation backend—SAM2 or SAM3—is right for your data.

---

## Choosing a Backend: SAM2 vs SAM3

SAM2 and SAM3 are not interchangeable — they work by completely different mechanisms. Understanding this distinction determines which workflow you follow.

=== "SAM2 — Exhaustive Proposal + Classifier Filter"

    SAM2 uses Automatic Mask Generation (AMG): it densely samples the image with a grid of point prompts and generates *every possible segment* it can find — organelles, contaminants, carbon, ice, everything with a detectable boundary. This candidate pool is then passed to a domain expert classifier that scores each mask and keeps only the target class.

    The model never decides what structure it is looking for — that decision belongs entirely to your classifier. This means **recall is bounded by boundary quality, not vocabulary**: if a structure has a clear boundary, SAM2 will propose it.

    **Use SAM2 when:**

    - You have trained (or plan to train) a domain expert classifier
    - You need reliable, reproducible results across a large dataset
    - You need 3D propagation through tomographic volumes
    - Your target structure has unusual appearance or contrast that text cannot reliably describe

    !!! tip "Zero-Shot Exploration"
        Without a classifier, SAM2 returns the raw, unfiltered candidate pool — every detectable boundary in the image. This is useful for assessing data quality before committing to annotation, but produces too many false positives for production use.

    ```python
    from saber.segmenters.micro import cryoMicroSegmenter
    from saber.adapters.base import SAM2AdapterConfig

    segmenter = cryoMicroSegmenter(cfg=SAM2AdapterConfig(cfg="large"))
    masks = segmenter.segment(image, target_class=1)
    ```

=== "SAM3 — Text-Guided Detection"

    SAM3 is a vision-language model that takes a plain-language description and *directly* detects matching structures — there is no AMG step and no candidate pool. The text prompt is the only signal the model uses to decide what to segment. No annotation, no classifier training, no preprocessing pipeline required.

    **Recall is bounded by language.** SAM3 can reliably detect structures it learned during pretraining (`"ribosome"`, `"mitochondria"`, `"membrane"`). Structures with unusual contrast, domain-specific appearance, or no common name may be missed or confused with visually similar objects.

    **Use SAM3 when:**

    - You need results immediately, before any annotation work
    - You are exploring a new dataset and want to quickly test segmentability
    - Your target is a well-known biological structure with a clear name
    - You cannot collect enough labeled examples for a classifier

    !!! warning "Not a drop-in replacement for SAM2 + classifier"
        SAM3 bypasses the entire SABER annotation and training workflow. It is faster to start but less reliable at scale on specialized cryo-ET data. For production pipelines, SAM2 with a trained classifier will almost always produce higher precision and recall.

    ```python
    from saber.segmenters.micro import cryoMicroSegmenter
    from saber.adapters.base import SAM3AdapterConfig

    segmenter = cryoMicroSegmenter(cfg=SAM3AdapterConfig())
    masks = segmenter.segment(image, text="ribosome")
    ```

??? info "Which model size should I use with SAM2?"
    SAM2 comes in four sizes: `tiny`, `small`, `base`, and `large`. Larger models produce higher-quality masks but require more GPU memory.

    | Size | GPU Memory | Recommended For |
    |------|-----------|-----------------|
    | `tiny` | ~2 GB | Fast iteration, CPU-limited machines |
    | `small` | ~4 GB | Default for most workflows |
    | `base` | ~6 GB | Better boundary precision |
    | `large` | ~10 GB | Publication-quality results, final runs |

    When a classifier config is loaded via `SAM2AdapterConfig(classifier=predictor)`, the model size is automatically read from the classifier's saved config—no need to set it manually.

---

## Core Architecture

<div class="grid cards" markdown>

-   :material-layers-outline: **Segmenter Classes**

    ---

    The main interface for running segmentation. Wraps the adapter and handles 2D/3D logic, sliding windows, and classifier filtering.

    `saber.segmenters.micro` · `saber.segmenters.tomo`

    [:octicons-arrow-right-24: 2D Quickstart](quickstart2d.md)
    [:octicons-arrow-right-24: 3D Quickstart](quickstart3d.md)

-   :material-swap-horizontal: **Adapter Configs**

    ---

    Thin configuration objects that select and configure the underlying model backend (SAM2 or SAM3). Passed into segmenter constructors.

    `SAM2AdapterConfig` · `SAM3AdapterConfig`

    [:octicons-arrow-right-24: SAM2 vs SAM3 choice](#choosing-a-backend-sam2-vs-sam3)

-   :material-brain: **Domain Expert Classifiers**

    ---

    Optional CNN classifiers trained on your annotated data. Loaded into `SAM2AdapterConfig` to filter false-positive masks at inference time.

    `saber.classifier.models.common`

    [:octicons-arrow-right-24: Training Guide](training.md)

-   :material-cog-outline: **Utility Modules**

    ---

    I/O, preprocessing, parallelization, and visualization helpers used across all workflows.

    `saber.utils.io` · `saber.utils.preprocessing`
    `saber.utils.parallelization` · `saber.visualization`

    [:octicons-arrow-right-24: Parallel Inference](parallel-inference.md)

</div>

---

## Quick Start Paths

=== "2D Micrographs"

    For single-particle EM, SEM, or any 2D image data.

    ```python
    from saber.segmenters.micro import cryoMicroSegmenter
    from saber.adapters.base import SAM2AdapterConfig
    from saber.utils import io

    image, pixel_size = io.read_micrograph("path/to/image.mrc")

    segmenter = cryoMicroSegmenter(
        cfg=SAM2AdapterConfig(cfg="large"),
        min_mask_area=50,
    )
    masks = segmenter.segment(image, target_class=1)
    ```

    [:octicons-arrow-right-24: Full 2D Quickstart](quickstart2d.md)

=== "3D Tomograms"

    For cryo-ET volumes, FIB-SEM, or any 3D data managed via Copick.

    ```python
    from saber.segmenters.tomo import tomoSegmenter
    from saber.adapters.base import SAM2AdapterConfig
    from copick_utils.io import readers

    vol = readers.tomogram(run, voxel_size=10, algorithm="denoised")

    segmenter = tomoSegmenter(cfg=SAM2AdapterConfig(cfg="large"))
    vol_masks = segmenter.segment(vol, thickness=10, target_class=1)
    ```

    [:octicons-arrow-right-24: Full 3D Quickstart](quickstart3d.md)

=== "With a Trained Classifier"

    Load a trained domain expert classifier to dramatically reduce false positives.

    ```python
    from saber.segmenters.micro import cryoMicroSegmenter
    from saber.adapters.base import SAM2AdapterConfig
    from saber.classifier.models import common

    # Classifier config carries the model size — no need to set cfg= manually
    predictor = common.get_predictor(
        model_weights="results/best_model.pth",
        model_config="results/model_config.yaml",
    )

    segmenter = cryoMicroSegmenter(
        cfg=SAM2AdapterConfig(classifier=predictor),
        min_mask_area=50,
    )
    masks = segmenter.segment(image, target_class=1)
    ```

    [:octicons-arrow-right-24: Training a Classifier](training.md)

---

## Output Formats

| Format | Use Case |
|--------|----------|
| **NumPy array** | Direct access for custom analysis and downstream processing |
| **Zarr volume** | Efficient on-disk storage for large datasets and 3D masks |
| **Copick segmentation** | Native collaborative annotation format for cryo-ET projects |
| **PNG gallery** | Visual review of batch segmentation results |
