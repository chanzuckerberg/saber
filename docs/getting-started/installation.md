# Installation Guide

## Requirements 

Saber runs on Python 3.10 and above on Linux or Windows with CUDA12. 

## Quick Installation

Saber is available on PyPI and can be installed using pip:

```bash
pip install saber-em
```

**⚠️ Note** By default, the GUI is not included in the base installation.
To enable the graphical interface for manual annotation, install with:
```bash
pip install saber-em[gui]
```

## Development Installation

If you want to contribute to saber or need the latest development version, you can install from source:

```bash
git clone https://github.com/czi-ai/saber.git
cd saber
pip install -e .
```

## Download Pre-trained Models

After installation, download the required SAM2 model weights:

```bash
saber download sam2-weights 
```
This downloads the model weights to `path/to/site-packages/saber/checkpoints/`

## Verification

To verify your installation, run:

```bash
python -c "import saber; print(saber.__version__)"
```

## Next Steps

- [Import Tomograms](import-tomos.md) - Learn how to import your tomograms into a copick project.
- [Quick Start Guide](quickstart.md) - Run your first 2D or 3D experiment. 
- [Learn the API](../api/quickstart.md) - Integrate Saber into your Python workflows. 