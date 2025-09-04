# Cryo-ET Data Import Guide

SABER leverages [copick](https://github.com/copick/copick) to provide a flexible and unified interface for accessing tomographic data, whether it's stored locally or remotely on HPC servers or on our [CryoET Data Portal](https://cryoetdataportal.czscience.com). This guide explains how to work with both data sources.

## Starting a New Copick Project

The copick configuration file points to a directory that stores all tomograms, coordinates, and segmentations in an overlay root. Generate a config file using the command line:

```bash 
copick config filesystem --overlay-root /path/to/overlay
```

For cryo-ET workflows involving protein 3D coordinate annotation, you can define biological objects during project creation:

```bash
copick config filesystem \
    --overlay-root /path/to/overlay \
    --objects ribosome,True,130,6QZP \
    --objects apoferritin,True,65 \
    --objects membrane,False
```

<details markdown="1">
<summary><strong>👀 Example Copick Config File (`config.json`)</strong></summary>

The resulting `config.json` file would look like this:

```json
{
    "name": "test",
    "description": "A test project description.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "ribosome",
            "is_particle": true,
            "label": 1,
            "radius": 130,
            "pdb_id": "6QZP"
        },
        {
            "name": "apoferritin",
            "is_particle": true,
            "label": 2,
            "radius": 65            
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 3
        }
    ],

    "overlay_root": "local:///path/to/overlay",
    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "local:///path/to/static",
    "static_fs_args": {
        "auto_mkdir": true
    }    
}
```
**Directory Structure:**

- **Overlay root:** Writable directory where new results can be added, modified, or deleted
- **Static root:** Read-only directory that never gets manipulated (frozen data)

**Path Types:**

- **Local paths:** `local:///path/to/directory`
- **Remote paths:** `ssh://server/path/to/directory`

The `copick config filesystem` command assumes local paths, but you can edit the config file to specify remote locations.

</details>

</details>

!!! info
    The `--objects` flag accepts 2-4 elements separated by commas:

    1. **Particle name** (required): e.g., `ribosome`
    2. **Is pickable** (required): `True` for particles, `False` for continuous segmentations
    3. **Particle radius** (optional): in Ångströms, e.g., `130`
    4. **PDB ID** (optional): reference structure, e.g., `6QZP`

This structure supports both particle picking for sub-tomogram averaging and broader 3D segmentation tasks. Our deep learning platform [Octopi 🐙](https://github.com/chanzuckerberg/octopi) is designed to train models from copick projects for:

- Object 3D localization and particle picking
- Volumetric segmentation of cellular structures
- General 3D dataset annotation and analysis

## Starting a Copick Project Linked to the Data Portal

Create a copick project that automatically syncs with the [CryoET Data Portal](https://cryoetdataportal.czscience.com):

```bash
copick config dataportal --dataset-id DATASET_ID --overlay-root /path/to/overlay
```

This command generates a config file that syncs data from the portal with local or remote repositories. You only need to specify the dataset ID and the overlay or static path - pickable objects will automatically be populated from the dataset.

**Benefits:**

- Automatically populates pickable objects from the dataset
- Seamless integration with portal data
- Combines remote portal data with local overlay storage

## Importing Local MRC Files

### Prerequisites

This workflow assumes:

- All tomogram files are in a flat directory structure (single folder)
- Files are in MRC format (`*.mrc`)

### Import Command

If you have tomograms stored locally in `*.mrc` format (e.g., from Warp, IMOD, or AreTomo), you can import them into a copick project:

```bash
copick add tomogram \
    --config config.json \
    --tomo-type denoised \
    --voxel-size 10 \
    --no-create-pyramid \
    'path/to/volumes/*.mrc'
```

<details markdown="1">
<summary><strong>Import Parameters Explained</strong></summary>

- `--config config.json`: Path to your copick configuration file
- `--tomo-type denoised`: Specifies the tomogram type (options: `raw`, `denoised`, `filtered`)
- `--voxel-size 10`: Sets voxel size in Ångströms (10 Å = 1 nm recommended)
- `--no-create-pyramid`: Skips pyramid generation for faster import
- `'path/to/volumes/*.mrc'`: Path to your MRC file(s) - supports wildcards

</details>

## Advanced Import Options

If your data doesn't meet the standard requirements (flat directory structure + MRC format), please refer to our [API Import Documentation](../api/import-tomos.md), which covers:

- Nested directory structures
- Different file formats (TIFF, HDF5, etc.)
- Custom import scripts
- Batch processing workflows

---

*For more detailed workflows, see our [User Guide](../tutorials/introduction.md) and [API Documentation](../api/overview.md).*