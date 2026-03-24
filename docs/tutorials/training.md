# Training a Domain Expert Classifier

The classifier is what turns SAM2's generic mask proposals into organelle-specific segmentations. It is a lightweight neural network trained on your annotated data that scores each SAM2 candidate — keeping lysosomes, discarding carbon contamination, and ignoring everything else you didn't label.

---

## Step 1: Prepare Training Data

### Merging Multiple Datasets

If you have annotations from multiple experiments, merge them before training. More diverse training data leads to a classifier that generalizes better across imaging sessions and conditions.

```bash
saber classifier merge-data \
    --inputs 24aug09b,training1.zarr \
    --inputs 24aug30a,training2.zarr \
    --inputs 24oct24c,training3.zarr \
    --output merged_training.zarr
```

??? note "`saber classifier merge-data` Parameters"
    | Parameter | Description |
    |-----------|-------------|
    | `--inputs` | Comma-separated `experiment_id,zarr_file_path` pair. Repeat for each source. |
    | `--output` | Output merged Zarr file path |

??? question "When should I merge datasets?"
    Merging is beneficial when:

    - You have data from **different imaging sessions** (grid preparations, microscopes)
    - Your target structures vary in **size or density** across experiments
    - You want to reduce overfitting to a single acquisition

    Merging is *not* needed if all your data comes from a single, homogeneous experiment.

### Splitting into Train / Validation

```bash
saber classifier split-data \
    --input merged_training.zarr \
    --ratio 0.8
```

This creates `merged_training_train.zarr` (80%) and `merged_training_val.zarr` (20%). The split is done at the run level so no single tomogram or micrograph appears in both sets.

??? note "`saber classifier split-data` Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `--input` | Labeled Zarr to split | required |
    | `--ratio` | Fraction of data used for training | `0.8` |

---

## Step 2: Train the Classifier

```bash
saber classifier train \
    --input merged_training_train.zarr \
    --validate merged_training_val.zarr \
    --num-classes 3
```

??? note "`saber classifier train` Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `--input` | Training Zarr | required |
    | `--validate` | Validation Zarr | required |
    | `--num-classes` | Number of biological classes **+ 1** for background | required |
    | `--num-epochs` | Number of training epochs | `75` |

### Training Outputs

All results are saved to `results/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Model weights at the best validation epoch |
| `model_config.yaml` | Architecture, hyperparameters, and AMG settings |
| `metrics.pdf` | Loss and accuracy curves across training |
| `per_class_metrics.pdf` | Per-class precision, recall, and F1 |

!!! tip "The config YAML carries everything"
    When you later pass `model_config.yaml` to a segmenter, it automatically sets the SAM2 model size and AMG parameters that were used during preprocessing. You don't need to re-specify them at inference time.

??? note "What happens during training?"
    1. SAM2 image embeddings are extracted for each mask crop (no fine-tuning of SAM2 itself)
    2. A lightweight classification head learns to map embeddings to your class labels
    3. Validation loss is monitored every epoch; the best checkpoint is saved
    4. Training stops early if validation performance plateaus

---

## Step 3: Evaluate Your Classifier

Generate predictions on a held-out Zarr to visually inspect results before running full inference:

```bash
saber classifier predict \
    --model-weights results/best_model.pth \
    --model-config results/model_config.yaml \
    --input training1.zarr \
    --output training1_predictions.zarr
```

??? note "`saber classifier predict` Parameters"
    | Parameter | Description |
    |-----------|-------------|
    | `--model-weights` | Path to `best_model.pth` |
    | `--model-config` | Path to `model_config.yaml` |
    | `--input` | Zarr to run predictions on |
    | `--output` | Output Zarr for predicted labels |

This produces a Zarr with predicted class labels and an HTML gallery with masks overlaid on the original images, color-coded by class.

## Step 3: Share or Reuse Your Model

Your trained model is fully portable — the `results/` directory contains everything needed to run inference on any machine:

```bash
# Share with a collaborator or copy to your compute cluster
rsync -r results/ user@cluster:/path/to/results/
```

To use on a new machine:

```bash
saber segment tomograms \
    --config config.json \
    --model-weights results/best_model.pth \
    --model-config results/model_config.yaml \
    --target-class 1
```

---

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-arrow-right-24: **Run Inference**](inference.md)

    Apply your trained classifier to new 2D and 3D datasets.

-   [:octicons-arrow-right-24: **API: Training Guide**](../api/training.md)

    Customize the training loop programmatically — custom architectures, loss functions, and data augmentation.

</div>
