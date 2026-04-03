# SABER — Segment Anything Based Electron Recognition

![Segmentation Examples](assets/segmentation_example.png)

**SABER** is an open-source platform for autonomous segmentation of organelles in cryo-electron tomography (cryo-ET) and electron microscopy (EM) datasets. It combines state-of-the-art foundational models with expert-driven classification to deliver reliable, scalable segmentations — from a single micrograph to an entire project.

---

## Why SABER?

<div class="grid cards" markdown>

-   :material-molecule: **Foundational model power**

    Zero-shot segmentation using SAM2 and SAM3 — no annotations required to get started.

-   :material-brain: **Expert-driven accuracy**

    Train lightweight classifiers on your own annotations to distinguish organelles from contaminants and artifacts.

-   :material-layers-triple: **2D and 3D**

    Segment single micrographs or propagate across full tomographic volumes.

-   :material-file-export: **Publication-ready output**

    Export instance segmentations, semantic maps, coordinates, and per-organelle statistics.

</div>

---

## Tutorials

### CLI

<div class="grid cards" markdown>

-   :octicons-terminal-24: **Pre-processing**

    Prepare your EM/cryo-ET datasets and annotate segmentations using the interactive GUI.

    [:octicons-arrow-right-24: Get started](tutorials/preprocessing.md)

-   :octicons-cpu-24: **Training a Classifier**

    Train a domain expert classifier on your annotations to filter SAM2 mask proposals.

    [:octicons-arrow-right-24: Train](tutorials/training.md)

-   :octicons-play-24: **Inference (2D & 3D)**

    Apply your trained classifier to generate segmentations across entire datasets.

    [:octicons-arrow-right-24: Run inference](tutorials/inference.md)

-   :octicons-filter-24: **Membrane Refinement**

    Enforce topological consistency across organelle and membrane segmentations.

    [:octicons-arrow-right-24: Refine](tutorials/membrane-refinement.md)

</div>

### Python API

<div class="grid cards" markdown>

-   :octicons-code-24: **API Overview**

    Comprehensive introduction to the SABER Python API.

    [:octicons-arrow-right-24: Read more](api/overview.md)

-   :octicons-image-24: **2D Quickstart**

    Segment 2D micrographs programmatically in a few lines of code.

    [:octicons-arrow-right-24: Get started](api/quickstart2d.md)

-   :octicons-stack-24: **3D Quickstart**

    Segment 3D tomograms and propagate across volumes.

    [:octicons-arrow-right-24: Get started](api/quickstart3d.md)

-   :octicons-people-24: **Training Guide**

    Customize the training loop — architectures, loss functions, and data augmentation.

    [:octicons-arrow-right-24: Customize](api/training.md)

</div>

---

## Getting Help

Open an issue on our [GitHub repository](https://github.com/chanzuckerberg/saber).
