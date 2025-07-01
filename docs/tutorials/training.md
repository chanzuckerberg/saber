# Training a Classifier

Once you've completed the preprocessing and annotation steps, it's time to train your domain expert classifier. This classifier learns to map SAM2's generic segmentations to your specific biological classes, creating an intelligent system that can automatically identify structures in new data.

---

## 📊 Preparing Your Training Data

### Working with Multiple Data Sources

In many research scenarios, you'll have annotated data from multiple sources - different experimental acquisitions, various imaging conditions, or separate copick projects. SABER allows you to combine these datasets for robust classifier training.

<details markdown="1">
<summary><strong>Why merge multiple datasets?</strong></summary>

Combining annotations from multiple data sources creates a more robust and generalizable classifier by:

- **Increasing data diversity**: Different experimental conditions and imaging parameters
- **Improving generalization**: Reduces overfitting to specific acquisition settings
- **Enhancing coverage**: More examples of edge cases and rare structures
- **Building robustness**: Better performance across different imaging modalities

</details>

### Merging Annotated Datasets

If you have multiple annotated zarr files from different sources, combine them into a single training dataset:

```bash
saber classifier merge-data \
    --inputs 24aug09b,training1.zarr \
    --inputs 24aug30a,training2.zarr \
    --inputs 24oct24c,training3.zarr \
    --output merged_training.zarr
```

**Input format**: Each `--inputs` flag takes a comma-separated pair: `experiment_id,zarr_file_path`. This preserves the source information while creating a unified dataset.

### Creating Training and Validation Splits

Split your dataset (merged or single) into training and validation sets for proper model evaluation:

```bash
saber classifier split-data \
    --input merged_training.zarr \
    --ratio 0.8  # 80% training, 20% validation
```

**Output**: This creates two files:

- `merged_training_train.zarr` - Training data (80%)
- `merged_training_val.zarr` - Validation data (20%)

---

## 🧠 Training Your Domain Expert Classifier

### Core Training Command

With your data prepared, train the classifier that will learn to identify your specific biological structures:

```bash
saber classifier train \
    --input merged_training_train.zarr \
    --validate merged_training_val.zarr \
    --num-classes 3  # Number of your biological classes + background
```

**Class counting**: The `--num-classes` should be your number of biological classes plus one for the background. For example:

- Binary classification → `--num-classes 2` (1 class + background)
- Multi-Classification: `carbon,lysosome,artifacts` → `--num-classes 4` (3 classes + background)

### Training Process

<details markdown="1">
<summary><strong>What happens during training?</strong></summary>

The training process involves:

1. **Feature extraction**: Using SAM2's pre-trained embeddings as rich feature representations
2. **Classifier learning**: A lightweight neural network learns to map these features to your biological classes
3. **Validation monitoring**: Performance is continuously evaluated on the validation set
4. **Model checkpointing**: Best-performing models are saved based on validation metrics
5. **Early stopping**: Training stops if performance plateaus to prevent overfitting

</details>

**Training outputs**: All results are saved in the `results/` directory:

- `best_model.pth` - Best performing model weights
- `model_config.yaml` - Model configuration and hyperparameters
- `metrics.pdf` and `per_class_metrics.pdf` - Plots of average and per class metrics during training.

## 🔍 Testing Your Trained Model

### Generate Predictions and Visual Gallery

Test your trained classifier and create a visual gallery to assess performance:

```bash
saber classifier predict \
    --model-weights results/best_model.pth \
    --model-config results/model_config.yaml \
    --input training1.zarr \
    --output training1_predictions.zarr
```

**What this produces**:

- **Prediction masks**: Semantic segmentation results stored in zarr format
- **Visual gallery**: HTML gallery with segmentations overlaid on original images
- **Class-specific colors**: Each biological class gets a unique color for easy identification

### Evaluating Results

The prediction output includes:
- **Quantitative metrics**: Accuracy, precision, recall, and F1-scores per class
- **Visual assessment**: Side-by-side comparisons of predictions vs. annotations
- **Error analysis**: Identification of common failure modes and edge cases

<details markdown="1">
<summary><strong>Interpreting your results</strong></summary>

**Good signs**:

- High accuracy (>85%) on validation data
- Consistent performance across different experimental conditions
- Clear, well-defined segmentation boundaries
- Accurate classification of challenging cases

**Warning signs**:

- Large gap between training and validation accuracy (overfitting)
- Poor performance on certain classes (class imbalance)
- Inconsistent results across different image types
- Blurry or imprecise segmentation boundaries

**Next steps if results are poor**:

- Add more diverse training examples
- Balance your class distribution
- Adjust training hyperparameters
- Consider additional data augmentation

</details>


## 🚀 What's Next?

Your trained classifier is now ready for production use! You can:

- [**Apply to new data**:](inference.md) Use your model for automated segmentation
- **Scale your analysis**: Process large datasets efficiently
- **Share your model**: Export for use by collaborators
- **Iterate and improve**: Add new data and retrain for better performance

**Model portability**: Your trained weights (`best_model.pth`) and configuration (`model_config.yaml`) can be shared with colleagues or used across different computing environments.

---

_Ready for production? Check out the [Inference & Segmentation](inference.md) tutorial to learn how to apply your trained model to new datasets!_