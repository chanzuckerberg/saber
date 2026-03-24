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
git clone https://github.com/chanzuckerberg/saber.git
cd saber
pip install -e .
```

## Verification

To verify your installation, run:

```bash
saber
```

You should see the following output:

```bash
 SABER ⚔️ -- Segment Anything Based Expert             
 Recognition.                                                
╭─ Options ──────────────────────────────────────────╮
│ --help  -h  Show this message and exit.            │
╰────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────╮
│ analysis    Post-processing analysis after         │
│             segmentation.                          │
│ classifier  Routines for training and evaluating   │
│             classifiers.                           │
│ gui         Saber GUI for annotating SAM2          │
│             segmentations with custom classes.     │
│ save        Save organelle coordinates and         │
│             statistics (size-distribution) from    │
│             segmentations.                         │
│ segment     Segment Tomograms and Micrographs with │
│             SABER.                                 │
│ web         SABER Annotation GUI Web Server        │
╰────────────────────────────────────────────────────╯
```

## Troubleshooting

### HuggingFace Authentication Error (GatedRepoError)

If you see the following error when running Saber:

```
GatedRepoError: 401 Client Error.
Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json.
Access to model facebook/sam3 is restricted. You must have access to it and be authenticated to access it. Please log in.
```

This means the SAM3 model weights are hosted in a gated HuggingFace repository that requires you to explicitly request access. To fix this:

1. Create or log in to your [HuggingFace account](https://huggingface.co).
2. Visit the [facebook/sam3](https://huggingface.co/facebook/sam3) model page and request access.
3. Once access is granted, authenticate your local environment:
   ```bash
   pip install huggingface_hub
   hf auth login
   ```
   When prompted, enter your HuggingFace access token (available at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
4. Re-run your Saber command — the model weights will now download automatically.

## Next Steps

- [Import Tomograms](import-tomos.md) - Learn how to import your tomograms into a copick project.
- [Quick Start Guide](quickstart.md) - Run your first 2D or 3D experiment. 
- [Learn the API](../api/quickstart.md) - Integrate Saber into your Python workflows. 