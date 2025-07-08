# SABER⚔️
**S**egment **A**nything **B**ased **E**lectron tomography **R**ecognition is a robust platform designed for autonomous segmentation of organelles from cryo-electron tomography (cryo-ET) or electron microscopy (EM) datasets. 

## Introduction
Leveraging foundational models, SABER enables segmentation directly from video-based training translated into effective 3D tomogram analysis. Users can utilize zero-shot inference with morphological heuristics or enhance prediction accuracy through data-driven training.

## 💫 Key Features
* 🔍 Zero-shot segmentation: Segment EM/cryo-ET data without explicit retraining, using foundational vision models.
* 🖼️ Interactive GUI for labeling: Intuitive graphical interface for manual annotation and segmentation refinement.
* 🧠 Expert-driven classifier training: Fine-tune segmentation results by training custom classifiers on curated annotations.
* 🧊 3D organelle segmentation: Generate volumetric segmentation masks across tomographic slices.

## 🚀 Getting Started

### Installation

Saber is available on PyPI and can be installed using pip:
```bash
pip install saber-em
```

⚠️ **Note**: 

- By default, the GUI is not included in the base installation.
To enable the graphical interface for manual annotation, install with:
```bash
pip install saber-em[gui]
```
- One of the current dependencies is currently not working with pip 25. To temporarily reduce the pip version, run:
```bash
pip install --upgrade "pip<25"
```

### Basic Usage
SABER provides a clean, scriptable command-line interface. Run the following command to view all available subcommands:
```
saber --help
```
We can begin by downloading the pre-trained SAM2 weights:
```
saber download sam2-weights
```

## 📚 Documentation

For detailed documentation, tutorials, CLI and API reference, visit our [documentation](https://czi-ai.github.io/saber/)


## 🤝 Contributing

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.