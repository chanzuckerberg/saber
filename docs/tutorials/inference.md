# Inference (2D & 3D)

The inference phase is where your SABER workflow pays off - applying trained models or zero-shot segmentation to generate high-quality 2D and 3D segmentations on new datasets. Whether you're using raw SAM2 capabilities or your custom-trained classifier, SABER provides flexible options for both semantic and instance segmentation.

## 🎯 Segmentation Modes

SABER offers two distinct approaches for inference, depending on whether you have a trained classifier:

### Expert-Guided Segmentation 
Apply your custom-trained classifier for domain-specific results:

- **Semantic segmentation**: Identify all learned biological classes simultaneously.
- **Instance segmentation**: Focus on specific objects of interest with individual instance masks.

<details markdown="1">
<summary><strong>Semantic vs Instance Segmentation</strong></summary>

**Semantic Segmentation** (`--target-class -1`):

- Assigns every pixel to a biological class (background, lysosome, carbon, etc.)
- Produces a single mask with different colors/values for each class
- Ideal for understanding tissue composition and spatial relationships
- Answers: "What types of structures are present and where?"

**Instance Segmentation** (`--target-class N` where N > 0):

- Identifies individual objects of a specific class
- Each object gets a unique instance ID
- Enables counting, size analysis, and individual object tracking
- Answers: "How many lysosomes are there and what are their properties?"

</details>

---

## 🖼️ 2D Micrograph Segmentation

#### Basic Usage

Segment individual images or entire folders of micrographs:

```bash
# Single micrograph
saber segment micrographs \
    --input path/to/image.mrc \
    --output segmentation_results.zarr

# Batch processing
saber segment micrographs \
    --input 'path/to/micrographs/*.mrc' \
    --output batch_segmentations.zarr
```

**Parameters**:

| Parameter | Description | Example  |
|-----------|-------------|---------|
| `--input` | Path to micrograph or folder with file extension | `'path/*.mrc'` |
| `--output` | Path to output Zarr file | `results.zarr` |
| `--model-config` | Path to classifier model config | `results/model_config.yaml` |
| `--model-weights` | Path to trained classifier weights | `results/best_model.pth` |
| `--target-class` | Target class (-1 for semantic, N>0 for instance) | `2` |
| `--target-resolution` | Target resolution in Angstroms |  `10` |


## 🧊 3D Tomogram Segmentation

#### Basic Usage

Generate 3D segmentations using copick-managed data:

```bash
saber segment tomograms \
    --config copick_config.json \
    --model-config results/model_config.yaml \
    --model-weights results/best_model.pth \
    --segmentation-name organelles \
    --target-class 2 
```

When no `--run-ids` are provided, SABER will segment the entire project and save the results under provided `--segmentation-name` and `--segmentation-session-id` flags (by default, the `user-id` will always be saber. )

**Parameters**:

| Parameter | Description | Example  |
|-----------|-------------|---------|
| `--config` | Path to Copick config file | `copick_config.json` |
| `--voxel-size` | Resolution of tomograms to process | `10` |
| `--tomo-alg` | Reconstruction algorithm to query | `denoised` |
| `--slab-thickness` | Thickness of slab for initial segmentation | `10` |
| `--segmentation-name` | Name of segmentation session | `organelles` |
| `--segmentation-session-id` | Session ID for segmentation mask | `1` |
| `--model-config` | Path to classifier model config | `results/model_config.yaml` |
| `--model-weights` | Path to trained classifier weights | `results/best_model.pth` |
| `--target-class` | Target class (-1 for semantic, N>0 for instance) | `2` |
| `--num-slabs` | Number of slabs to segment | 0 |
| `--run-ids` | Specific tomogram runs to process | None | `Position_10_Vol,Position_15_Vol` |

---

_🎉 Congratulations! You now have the complete SABER workflow - from raw data to production-ready segmentations. For advanced usage and API integration, check out the [Python API Documentation](../api/quickstart.md)._