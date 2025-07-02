# Pre-processing Your Data

Pre-processing is the first step in the SABER workflow, where you'll prepare your electron microscopy (EM) or cryo-electron tomography (cryo-ET) data for segmentation and annotation. This step leverages [SAM2 (Segment Anything Model 2)](https://ai.meta.com/sam2/), a foundation model that can segment arbitrary structures without domain-specific training, making it ideal for scientific imaging applications.

---

## 🗂️ Supported File Types

SABER can read and process the following file formats:

- **MRC** (`.mrc`)
- **TIFF** (`.tif`, `.tiff`)
- **Zarr** (`.zarr`)

**Material science formats:**  

  - **SER** (`.ser`)
  - **DM3/DM4** (`.dm3`, `.dm4`)
- ...and more!

---

## 🔍 Quick Assessment: Exploring Raw SAM2 Segmentations

**Foundation Model Power**: SAM2 is a breakthrough foundation model developed by Meta AI that can segment objects across diverse domains with remarkable zero-shot capabilities. Despite being trained primarily on natural images, SAM2 can identify and delineate structures in electron microscopy data without any domain-specific training.

Before committing to the full SABER workflow, quickly assess whether your data works well with SAM2's segmentation capabilities:

### For Micrographs (2D Data)

For micrograph (2D data), we have the option to either directly segment the full image or downsample the data prior to running segmentation. `*.mrc` files or other file formats where the metadata is available we can provide the `--target-resolution` flag in Angstrom.

```bash
saber segment micrograph \
    --input path/to/image.mrc \
    --target-resolution 10 # Angstroms
```
For `*.tif` images, or data where the pixel size is unavailable in the meta-data, we need to provid the scale that we want to downsample our data.

```bash
saber segment micrograph \
    --input path/to/image.mrc \
    --scale 3 # Downsample by factor of 3
```

### For Tomograms (3D Data)

SABER uses copick to access tomographic volumes. You have two options for assessment:

**Single slab assessment** (quick preview at a central z-depth):
```bash
saber segment slab \
    --config config.json \
    --voxel-size 10 --tomo-alg denoised --slab-thickness 10 \
    --run-id Position_10_Vol
```

**Full tomogram segmentation** (segment the entire 3D tomgoram):
```bash
saber segment tomograms \
    --config config.json \
    --voxel-size 10 --tomo-alg denoised --slab-thickness 10 \
    --run-id Position_10_Vol
```

SABER's tomographic workflow begins with 2D segmentation initialization, then propagates segments into 3D. The --slab-thickness flag (in voxels) averages density within a specified thickness, improving segmentation quality by reducing noise. If structures segment well in 2D slabs, they typically translate well to full 3D segmentation.

This preview helps you understand what structures SAM2 naturally identifies in your data and plan your annotation strategy accordingly.

## 🧬 Pre-processing Electron Micrographs

For single-particle datasets, ADF/BF signals from S/TEM, or FIB-SEM micrographs -- use the `saber classifier prepare-micrograph-training` command:

```bash
saber classifier prepare-micrograph-training \
    --config config.json \
    --input 'path/to/*.mrc' \
    --output training.zarr
```

**Dense segmentation approach**: This command performs comprehensive segmentation across all input images, identifying every discernible structure that SAM2 can detect. The result is a complete inventory of potential biological features ready for expert classification.

---

## 🧩 Generating Training Data with Initial SAM2 Segmentations

<details>
<summary><strong>💡 Why SABER's Preprocessing is Different</strong></summary>

Traditional workflows require you to manually draw every mask from scratch. SABER precomputes ALL possible segments using SAM2's foundation model, then lets you focus on the science -- simply assigning biological meaning to structures that are already perfectly segmented.

</details>

Generate comprehensive slab-based segmentations that maintain 3D context:
```bash
saber classifier prepare-tomogram-training \
    --config config.json \
    --zarr-path output_zarr_fname.zarr \
    --num-slabs 3
```

This will save multiple slab-wise segmentations across different z-depths, stored in a Zarr volume format that preserves both the original data and comprehensive SAM2 masks. This multi-depth approach ensures comprehensive coverage of all potential biological features for expert annotation. 

![SABER GUI](../assets/multi_slab.png)

<details markdown="1">
<summary><strong>💡 Why multiple slabs?</strong></summary>
Small objects or sparse structures might not be present in a single slab projection. By generating multiple 2D slab projections at different z-depths, SABER captures as many segmentations and instances of your target objects as possible. This is particularly important for:

- Small organelles that appear sporadically through the volume
- Thin structures that might be missed in thick slab averages  
- Objects with variable density that become more visible at certain depths
</details>

---

## 🎨 Next Step: Annotation with the SABER GUI

Once preprocessing is complete, SABER's unique annotation workflow begins. Instead of drawing masks from scratch, you simply:

1. **Point and Click** on the precomputed segmentations.
2. **Assign Class Labels** using the dropdown menu.

![SABER GUI](../assets/saber_gui.png)

```bash
saber gui \
    --input output_zarr_fname.zarr \
    --output curated_labels.zarr \
    --class-names carbon,lysosome,artifacts
```

**Class Configuration**: The `--class-names` flag defines the biological classes present in your data. For binary classification (object vs. background), you can omit this flag for a simple two-class system.

**💡 How Many Micrographs / Tomograms Should I Annotate?** In general we recommend annotating 20-40 runs per dataset. In cases where there are several objects per image/slab the lower range should be sufficient. If only a few instances are available, the higher range is recommended.  

**Tip:** For transferring data between machines, it's recommended to compress your Zarr files:
```bash
zip -r curated_labels.zarr.zip curated_labels.zarr
```

---

_Ready to move on? Check out the [Training a Classifier](training.md) tutorial!_
