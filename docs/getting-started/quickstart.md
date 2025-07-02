# Quick Start Guide

This guide walks you through a complete Saber workflow: from curation of training data to segmenting micrographs or tomograms in both 2D and 3D.

## Basic Workflow Overview

The SABER workflow consists of three main phases:

1. **Data Preparation & Annotation** - Generate initial segmentations and curate training labels
2. **Model Training** - Train a domain expert classifier on your annotated data
3. **Production Inference** - Apply your trained model to new data

For reference, you can skip steps 1 and 2 to visualize the raw SAM2 segmentations in 2D or 3D without a domain expert classifier. 

## 🧩 Phase 1: Curating Training Labels and Training and Domain Expert Classifier 

### Producing Intial SAM2 Segmentations
Use `prepare-tomogram-training` to generate 2D segmentations from a tomogram using SAM2-style slab-based inference. These masks act as a rough initialization for downstream curation and model training.

#### For tomogram data:
```bash
saber classifier prepare-tomogram-training \
    --config config.json \
    --voxel-size 10 --tomo-alg denoised \
    --num-slabs 3 --output training_data.zarr \
```
This will save slab-wise segmentations in a Zarr volume that can be reviewed or refined further.

#### For electron micrograph/single-particle data:
```bash
saber classifier prepare-micrograph-training \
    --input path/to/folder/*.mrc \
    --ouput training_data.zarr \
    --target-resolution 10 
```

In the case of referencing MRC files from single particle datasets use `prepare-micrograph-training` instead. 

### 🎨 Annotating Segmentations for the Classifier with the Interactive GUI

Launch an interactive labeling session to annotate the initial SAM2 segmentations and assign class labels.
```
saber gui \
    --input output_zarr_fname.zarr \
    --output curated_labels.zarr \
    --class-names carbon,lysosome,artifacts
```

For transfering the data between machines, its recommended ziping (compressing) the zarr file prior to data transfer (e.g. `zip -r curated_labels.zarr.zip curated_labels.zarr`).

Once annotations are complete, split the dataset into training and validation sets:

```
saber classifier split-data \
    --input curated_labels.zarr \
    --ratio 0.8
```
This generates `curated_labels_train.zarr` and `curated_labels_val.zarr` for use in model training.

## 🧠 Phase 2: Train a Domain Expert Classifier

Train a classifier using your curated annotations. This model improves segmentation accuracy beyond zero-shot results by learning from expert-provided labels.
```
saber classifier train \
    --train curated_labels_train.zarr --validate curated_labels_val.zarr \
    --num-epochs 75 --num-classes 4 
```
The number of classes should be 1 greater than the number of class names provided during annotation (to account for background).
Training logs, model weights, and evaluation metrics will be saved under `results/`.

## 🔍 Phase 3: Inference

### 🖼️ Producting 2D Segmentations with SABER

SABER operates in two modes depending on your input: interactive mode when processing a single image, and batch processing mode when you provide a file path pattern (like `--input 'path/to/*.mrc'`) to process entire datasets automatically.

```bash
saber segment micrographs \
    --input path/to/image.mrc \
    --model-config results/model_config.yaml \
    --model-weights results/best_model.pth \
    --target-resolution 10 # Angstrom 
    # provide --scale 3 instead if you want to dowsample by 3
```

### 🧊 Producing 3D Segmentations with SABER 

For tomograms, SABER enters interactive mode when you specify particular `--run-ids`, or batch processes the entire project when `--run-ids` is omitted.

```bash
saber segment tomograms \
    --config config.json
    --model-config results/model_config.yaml \
    --model-weights results/best_model.pth \
    --target-class 2 --run-ids Position_12_Vol
```

## What's Next?
This workflow gives you a quick introduction to the saber segmentation pipeline. To learn more:

* [Checkout the tutorial](../tutorials/introduction.md).
* [Learn how to use the API](../api/overview.md). 