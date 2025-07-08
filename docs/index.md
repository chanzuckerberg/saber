# SABER⚔️ — Segment Anything Based Electron Tomography Recognition

![Segmentation Examples](assets/segmentation_example.png)

Welcome to the documentation for **SABER**, a robust, open-source platform for autonomous segmentation of organelles in cryo-electron tomography (cryo-ET) and electron microscopy (EM) datasets.

SABER leverages state-of-the-art foundational models to enable:

- **Zero-shot segmentation** of EM/cryo-ET data
- **Interactive annotation** with a user-friendly GUI
- **Expert-driven classifier training** for improved accuracy
- **3D organelle reconstruction** from tomographic data

---

## 🚀 What Can You Do With SABER?

- **Pre-process and curate** your microscopy data for segmentation
- **Train custom classifiers** using your own expert annotations
- **Run inference** for both 2D and 3D segmentation tasks
- **Visualize and refine** results interactively

---

## 💡 Why SABER?

SABER is designed for researchers who want to:

- Accelerate organelle segmentation in large EM/cryo-ET datasets
- Combine the power of foundational models with expert curation
- Seamlessly move from raw data to publication-ready segmentations in both 2D and 3D.

---

## 📚 Tutorials

SABER can be used both from the command line (CLI) and as a Python library (API). Choose the workflow that fits your needs. Follow the [quick start guide](getting-started/quickstart.md) to learn the core commands to develop segmentations. 

### 🖥️ CLI Tutorials
Use SABER from the command line for quick, scriptable workflows:

- [Pre-processing Your Data](tutorials/preprocessing.md): Prepare your EM/cryo-ET datasets for segmentation and annotation. Use the interactive GUI to annotate segmentations and 
- [Training a Classifier](tutorials/training.md): With the annotations, train a domain expert classifier.
- [Inference in 2D & 3D](tutorials/inference.md): Apply your trained models to generate high-quality segmentations in both 2D and 3D.
- [Refine Membranes](tutorials/membrane-refinement.md): Clean up organelle and membrane segmentations using GPU-optimized morphological filtering.

### 🐍 API Tutorials
Integrate SABER into your own Python scripts and notebooks:

- **[API Overview](api/overview.md)**: Comprehensive introduction to the SABER Python API
- **[2D Quickstart](api/quickstart2d.md)**: Learn how to segment 2D micrographs programmatically
- **[2D Workflow Tutorial](api/micrograph-workflow.md)**: Complete 2D pipeline with preprocessing, training, and inference
- **[3D Quickstart](api/quickstart3d.md)**: Learn how to segment 3D tomograms programmatically
- **[3D Workflow Tutorial](api/volume-workflow.md)**: Complete 3D pipeline with video propagation and advanced features

---

## 🙋‍♂️ Getting Help

Visit our [GitHub repository](https://github.com/czi-ai/segment-microscopy-sam2) for source code and issues.

---

_Ready to get started? Check out the [Quick Start](getting-started/quickstart.md) or the [API Overview](api/overview.md)!_

