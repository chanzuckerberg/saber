# Installation Guide

## Requirements 

copick runs on Python 3.10 and above on Linux or Windows with CUDA12. 

## Quick Installation

Saber is available on PyPI and can be installed using pip:

```bash
pip install saber
```

## Development Installation

If you want to contribute to octopi or need the latest development version, you can install from source:

```bash
git clone https://github.com/czi-ai/segment-microscopy-sam2.git
cd saber
pip install -e .
```

## Download the SAM2 Model Weights

After saber is available, download the pre-trained SAM2 model weights with the following command:

```bash
saber download sam2-weights 
```

## Verification

To verify your installation, run:

```bash
python -c "import saber; print(saber.__version__)"
```

## Next Steps

- [Import Tomograms](import-tomos.md) - Learn how to import your tomograms into a copick project.
- [Quick Start Guide](quickstart.md) - Run your first 2D or 3D experiment. 
- [Learn the API](../api/quickstart.md) - Integrate Octopi into your Python workflows. 