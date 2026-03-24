# Inference

This page covers running SABER on new data — from a single image to an entire Copick project. SABER supports two modes depending on whether you have a trained classifier, and two modalities depending on whether your data is 2D or 3D.

---

## Segmentation Modes

SABER's two backends work fundamentally differently. Choosing between them determines whether you need to annotate data and train a classifier at all.

=== "SAM2 + Classifier (Recommended)"

    SAM2 runs Automatic Mask Generation (AMG), which densely samples the image and proposes *every* detectable segment — organelles, contaminants, ice, artifacts, everything. A trained domain expert classifier then scores each candidate and keeps only the class you care about.

    **Recall is bounded by boundary quality, not language.** If a structure has a clear boundary, SAM2 will find it. The classifier decides what it is.

    !!! info "Instance vs. semantic segmentation"
        - **`--target-class N`** (N > 0): Each organelle gets a unique instance ID. Use this for counting, sizing, and coordinate export.
        - **`--target-class -1`**: Every pixel is assigned to its most likely class. Use this for full tissue maps.

=== "SAM2 Zero-Shot (no classifier)"

    SAM2 AMG runs as above — proposing every possible segment — but with no classifier to filter results. You see the raw, unfiltered output of the model. This is useful for assessing whether your structures are cleanly detectable before committing to annotation.

    **The output will contain many false positives.** Every detectable boundary in the image gets a mask, regardless of biological relevance.

    !!! tip "Use this before annotating"
        If your target structures are clearly outlined in zero-shot results, annotation will be fast and the classifier will converge quickly. If they're fragmented or missing, investigate resolution, slab thickness, or SAM2 model size first.

    !!! warning "Not suitable for batch runs"
        Without a classifier, there is no way to distinguish organelles from contaminants at scale. Reserve zero-shot SAM2 for interactive quality assessment only.

=== "SAM3 — Text Prompt"

    SAM3 works differently from SAM2. Rather than proposing everything and filtering, it uses a vision-language model to *directly* detect structures matching a text description. There is no AMG step, no candidate pool, and no classifier — the text prompt is the only signal.

    **Recall is bounded by language.** SAM3 can only find structures it can recognize from its training vocabulary. Well-known structures (`"ribosome"`, `"mitochondria"`) work well. Unusual structures, artifacts, or anything domain-specific may be missed or confused with visually similar objects.

    !!! info "SAM3 bypasses Stages 1–2 of the SABER workflow"
        You do not need to annotate data or train a classifier to use SAM3. It goes directly from raw image to segmentation. This makes it the fastest path to results on a new dataset, but the least reliable for production batch processing.

---

## 2D Micrograph Segmentation

=== "With Classifier"

    ```bash
    saber segment micrographs \
        --input 'path/to/micrographs/*.mrc' \
        --output segmentation_results.zarr \
        --model-config results/model_config.yaml \
        --model-weights results/best_model.pth \
        --target-class 1
    ```

    When a glob pattern is provided, SABER automatically batches across all matching files. When a single file is given, it opens an interactive preview.

=== "SAM2 Zero-Shot"

    No classifier — SAM2 proposes every detectable segment. Use interactively to assess data quality before annotating.

    ```bash
    saber segment micrographs \
        --input path/to/image.mrc \
        --target-resolution 10
    ```

=== "SAM3 — Text Prompt"

    No classifier, no annotation required. SAM3 directly detects structures matching the text description. Use for fast exploration; not recommended for production batch runs.

    ```bash
    saber segment micrographs \
        --input path/to/image.mrc \
        --text-prompt "ribosome" \
        --target-resolution 10
    ```

??? note "`saber segment micrographs` Parameters"
    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | `--input` | File or glob pattern | `'path/*.mrc'` |
    | `--output` | Output Zarr | `results.zarr` |
    | `--model-config` | Classifier config YAML | `results/model_config.yaml` |
    | `--model-weights` | Classifier weights | `results/best_model.pth` |
    | `--target-class` | Class index (-1 for semantic) | `1` |
    | `--target-resolution` | Downsample target in Å (MRC) | `10` |
    | `--scale` | Downsample factor (TIF / no metadata) | `2` |

---

## 3D Tomogram Segmentation

=== "Batch (All Runs)"

    Omitting `--run-ids` processes every run in the Copick project and saves results under `--seg-name`:

    ```bash
    saber segment tomograms \
        --config copick_config.json \
        --model-config results/model_config.yaml \
        --model-weights results/best_model.pth \
        --seg-name organelles \
        --target-class 1
    ```

    !!! tip "seg-name and session ID"
        Results are stored in Copick under `--seg-name` / `--seg-session-id`. The default `--user-id` is always `saber`. If you re-run with different settings, increment `--seg-session-id` to avoid overwriting prior results.

=== "Interactive (Single Run)"

    Providing one or more `--run-ids` triggers interactive mode — results are displayed immediately rather than saved:

    ```bash
    saber segment tomograms \
        --config copick_config.json \
        --model-config results/model_config.yaml \
        --model-weights results/best_model.pth \
        --target-class 1 \
        --run-ids Position_12_Vol
    ```

=== "Multi-Slab"

    For sparse or small structures, segment at multiple Z-depths and merge results using `--multi-slab`:

    ```bash
    saber segment tomograms \
        --config copick_config.json \
        --model-config results/model_config.yaml \
        --model-weights results/best_model.pth \
        --seg-name organelles \
        --target-class 1 \
        --multi-slab 10,30    # (1)
    ```

    1. Format is `thickness,spacing` — thickness of each slab in voxels, and the Z-spacing between initialization depths. Omitting the spacing uses a default of 30 voxels.

??? note "`saber segment tomograms` Parameters"
    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | `--config` | Copick config file | `copick_config.json` |
    | `--voxel-size` | Tomogram resolution in Å | `10` |
    | `--tomo-alg` | Reconstruction type | `denoised` |
    | `--slab-thickness` | Z-slab thickness in voxels | `10` |
    | `--seg-name` | Output segmentation name in Copick | `organelles` |
    | `--seg-session-id` | Session ID for versioning results | `1` |
    | `--model-config` | Classifier config YAML | `results/model_config.yaml` |
    | `--model-weights` | Classifier weights | `results/best_model.pth` |
    | `--target-class` | Class index (-1 for semantic) | `1` |
    | `--multi-slab` | Multi-depth config: `thickness` or `thickness,spacing` | `10,30` |
    | `--run-ids` | Specific runs to process (default: all) | `Position_10_Vol` |

---

## Exporting Statistics

After segmentation, extract per-organelle statistics (volume, diameter, coordinates) as a CSV:

```bash
saber analysis statistics \
    --config copick_config.json \
    --organelle-name organelles \
    --voxel-size 10
```

??? note "`saber analysis statistics` Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `--config` | Copick config file | required |
    | `--organelle-name` | Segmentation name to measure | required |
    | `--voxel-size` | Voxel size in Å for unit conversion | `10` |
    | `--session-id` | Session ID of the segmentation to read | — |
    | `--run-ids` | Specific runs to process (default: all) | — |

---

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-arrow-right-24: **Membrane Refinement**](membrane-refinement.md)

    If you segmented both organelles and membranes, use this to enforce topological consistency.

-   [:octicons-arrow-right-24: **API Overview**](../api/overview.md)

    Integrate SABER into your own Python pipelines.

</div>
