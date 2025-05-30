# SABER⚔️
**S**egment **A**nything **B**ased **E**lectron tomography **R**ecognition is a robust platform designed for autonomous segmentation of organelles from cryo-electron tomography (cryo-ET) or electron microscopy (EM) datasets. Leveraging foundational models, SAM2-ET enables segmentation directly from video-based training translated into effective 3D tomogram analysis. Users can utilize zero-shot inference with morphological heuristics or enhance prediction accuracy through data-driven training.

## 💫 Key Features
* 🔍 Zero-shot segmentation: Segment EM/cryo-ET data without explicit retraining, using foundational vision models.
* 🖼️ Interactive GUI for labeling: Intuitive graphical interface for manual annotation and segmentation refinement.
* 🧠 Expert-driven classifier training: Fine-tune segmentation results by training custom classifiers on curated annotations.
* 🧊 3D organelle reconstruction: Generate volumetric segmentation masks across tomographic slices.

## 🚀 Getting Started

To this package, it is recommended to have at a minumum Cuda 12.4 driver installed. In the case on bruno, you can load this module and build a conda environment with the following commands. 
`ml cuda/12.6.3_560.35.05 cudnn/8.9.7.29_cuda12`
`conda create --prefix=pySAM2 python=3.10`

To install the interactive labeling GUI, run the following command:
`pip install -e ".[gui]"`

**Note**: PyPI installation support coming soon.

SABER provides a clean, scriptable command-line interface. Run the following command to view all available subcommands:
```
saber --help
```
We can begin by downloading the pre-trained SAM2 weights:
```
saber download sam2-weights
```

## 🧪 Example Usage

### Curating Training Labels and Training and Domain Expert Classifier 

#### 🧩 Producing Intial SAM2 Segmentations
Use `prepare-tomogram-training` to generate 2D segmentations from a tomogram using SAM2-style slab-based inference. These masks act as a rough initialization for downstream curation and model training.

```
classifier prepare-tomogram-training \
    --config config.json \
    --zarr-path output_zarr_fname.zarr \
    --num-slabs 3
```
This will save slab-wise segmentations in a Zarr volume that can be reviewed or refined further.
In the case of referencing MRC files from single particle datasets use `prepare-micrograph-training` instead. 

#### 🎨 Annotating Segmentations for the Classifier with the Interactive GUI

Launch an interactive labeling session to annotate the initial SAM2 segmentations and assign class labels.
```
saber-gui \
    --input output_zarr_fname.zarr \
    --output curated_labels.zarr \
    --class-names carbon,lysosome,artifacts
```

For transfering the data between machines, its recommended ziping (compressing) the zarr file prior to data transfer (e.g. `zip -r curated_labels.zarr.zip curated_labels.zarr`).

Once annotations are complete, split the dataset into training and validation sets:

```
classifier split-data \
    --input curated_labels.zarr \
    --train-split 0.8
```
This generates `curated_labels_train.zarr` and `curated_labels_val.zarr` for use in model training.

#### 🧠 Train a Domain Expert Classifier

Train a classifier using your curated annotations. This model improves segmentation accuracy beyond zero-shot results by learning from expert-provided labels.
```
classifier train \
    --train curated_labels_train.zarr --validate curated_labels_val.zarr \
    --num-epochs 75 --num-classes 4 
```
The number of classes should be 1 greater than the number of class names provided during annotation (to account for background).
Training logs, model weights, and evaluation metrics will be saved under `results/`.

### 🔍 Inference

#### 🖼️ Producting 2D Segmentations with SABER

-- TODO -- 

#### 🧊 Producing 3D Segmentations with SABER 

Use the trained model to generate volumetric segmentations across the entire tomogram. The segment command supports slab-based 3D inference for smoother, context-aware outputs.
```
segment tomograms \
    --config config.json
    --model-config results/model_config.yaml \
    --model-weights results/best_model.pth \
    --target-class 2 --num-slabs 3 --segmentation-name lysosome
```
Saves a 3D mask labeled as lysosome.
`--target-class` must match the class index from the annotation step.

## Contact

For questions, feature requests, or collaboration inquiries, contact:

email: [jonathan.schwartz@czii.org](jonathan.schwartz@czii.org)
