# User Guide Overview

Welcome to the SABER User Guide! This tutorial series will take you from raw microscopy data to precise feature segmentations using foundational models and expert-driven training.

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
- **[Explore the API →](../api/quickstart.md)** - For programmatic usage

### ⚡ **Want immediate results?**
- **[Try the Quick Start →](quickstart.md)** - See SABER in action in 30 minutes

---

*Each tutorial builds on the previous ones, but you can jump to specific sections based on your needs and existing progress.*