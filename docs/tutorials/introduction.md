# User Guide Overview

Welcome to the SABER User Guide! This tutorial series will take you from raw microscopy data to precise feature segmentations using foundational models and expert-driven training.

## 🔬 How SABER Works

SABER is a few-shot segmentation framework that enables large-scale annotation of tomographic datasets with minimal manual input. By combining SAM2's general-purpose segmentation capabilities with a lightweight classifier tailored to microscopy-specific context, SABER produces rich contextual annotations efficiently.

![SABER Workflow](../assets/workflow.png)

#### Three-Stage Process

**Stage 1: Foundation Model Segmentation** - SABER begins by extracting 2D slices from your 3D tomogram (or processing 2D micrographs directly). SAM2's automatic mask generation identifies and segments every discernible structure in the image, creating comprehensive semantic segmentations without requiring any domain-specific training. This foundation model approach captures cellular structures, organelles, and artifacts that traditional segmentation methods might miss, providing a rich starting point for expert annotation.

**Stage 2: Expert-Guided Classification** - A domain expert classifier learns from user input annotations how to map SAM2's structure-agnostic segments to specific biological classes (organelles, membranes, artifacts, etc.). This creates context-aware segmentations tailored to your research needs, shown as the refined purple regions with expert-identified features.

**Stage 3: 3D Propagation of Prompts** For tomographic data, the trained classifier generates semantic masks on adjacent 2D slices, which serve as prompts for SAM2's video segmentation capabilities. This enables coherent propagation of segmentations through the z-stack (z₀₋₁, z₀, z₀₊₁), maintaining temporal consistency as structures evolve across slices. The result is a complete 3D reconstruction of your segmented features.

*Note: For 2D micrograph workflows, Stage 3 is optional - you can stop after obtaining high-quality 2D segmentations.*

## Tutorial Sections

### 🗂️ [Data Preprocessing](preprocessing.md)
**Prepare your microscopy datasets for segmentation**

Learn how to import and prepare your data for SABER workflows:

- Format conversion and data validation for EM, SEM, S/TEM, and SEM-FIB data
- Generate initial SAM2-based segmentations
- Integration with existing data management systems

**When to use:** Start here with raw microscopy data before any segmentation work.

### 🧠 [Training & Annotation](training.md)
**Create expert annotations and train custom classifiers**

Master the complete training workflow from initial segmentation to custom models:

- Interactive GUI annotation and curation
- Domain expert classifier training for organelles, nanoparticles, or custom features
- Model evaluation and optimization

**When to use:** After preprocessing, use this to create accurate models for your specific features and datasets.

### 🔍 [Inference & Segmentation](inference.md)
**Apply trained models to generate 2D and 3D feature segmentations**

Deploy your models to analyze new data:

- Zero-shot segmentation with foundational models
- Custom model inference for 2D micrographs and 3D tomograms
- Batch processing and performance optimization
- Quality control and result validation

**When to use:** Once you have trained models, use this to segment new datasets at scale.

## What's Next?

Ready to start? Choose your entry point:

### 🚀 **New to SABER?**
Follow the complete workflow:

- **[Begin with Preprocessing →](preprocessing.md)** - Start with raw data preparation
- **[Jump to Training →](training.md)** - If your data is already formatted

### 🔬 **Have existing models?**
- **[Skip to Inference →](inference.md)** - If you have pre-trained classifiers

### 🐍 **Python developer?**
- **[Explore the API →](../api/overview.md)** - For programmatic usage

### ⚡ **Want immediate results?**
- **[Try the Quick Start →](../getting-started/quickstart.md)** - See SABER in action in 30 minutes

---

*Each tutorial builds on the previous ones, but you can jump to specific sections based on your needs and existing progress.*