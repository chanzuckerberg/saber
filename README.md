# SABER⚔️

[![License](https://img.shields.io/pypi/l/saber-em.svg?color=green)](https://github.com/chanzuckerberg/saber/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/saber-em.svg?color=green)](https://pypi.org/project/saber-em/)
[![Python Version](https://img.shields.io/pypi/pyversions/saber-em.svg?color=green)](https://www.python.org/)
[![Manuscript](https://img.shields.io/badge/Manuscript-Patterns-green)](https://doi.org/10.1016/j.patter.2026.101688)

**S**egment **A**nything **B**ased **E**xpert **R**ecognition is a robust platform designed for autonomous segmentation of organelles from cryo-electron tomography (cryo-ET) or electron microscopy (EM) datasets. 

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
- One of the current dependencies is currently not working with pip 25.1. We recommend using pip 25.2 or higher when installing saber:
```bash
pip install --upgrade "pip>=25.2"
```

### Basic Usage
SABER provides a clean, scriptable command-line interface. Run the following command to view all available subcommands:
```
saber --help
```

## 📚 Documentation

For detailed documentation, tutorials, CLI and API reference, visit our [documentation](http://chanzuckerberg.github.io/saber/)

## 📄 Citation

If you use octopi in your research, please use the following BibTeX entry.

```bibtex
@article{schwartz2026saber,
  title     = {Large-scale CryoET segmentation from minimal expert annotations with foundation video models},
  author    = {Schwartz, Jonathan and Ermel, Utz and Kandel, Saugat and Siems, Hannah and Woldeyes, Rahel and Peukes, Julia and Cao, Ruiming and Andrade Lopez, Jose Miguel and Hoogenraad, Casper C. and Sheu, Shu-Hsien and Zheng, Shawn and Serwas, Daniel and Kimanius, Dari},
  journal   = {Patterns},
  year      = {2026},
  doi       = {10.1016/j.patter.2026.101688}
}
```

## 🤝 Contributing

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. 
Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.
