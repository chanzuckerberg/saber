# Data Import Guide

SABER leverages [copick](https://github.com/copick/copick) to provide a flexible and unified interface for accessing tomographic data, whether it's stored locally or remotely on a HPC server or on our [CryoET Data Portal](https://cryoetdataportal.czscience.com). This guide explains how to work with both data sources. If you need help creating these configuration files, detailed tutorials are available:

- [Copick Quickstart](https://copick.github.io/copick/quickstart/) - Basic configuration and setup 
- [Data Portal Tutorial](https://copick.github.io/copick/examples/tutorials/data_portal/) - Working with CryoET Data Portal

## Starting a New Copick Project

The copick configuration file points to a directory that stores all the tomograms, coordinates, and segmentations in an overlay root. We can generate a config file through the command line with `copick filesystem`. An example command would be:

```bash
copick config filesystem \
    --overlay-root /path/to/overlay --objects ribosome,True,130,6QZP \
    --objects apoferritin,True,65 --objects membrane,False
```

<details>
<summary><strong>💡 Example Copick Config File (config.json) </strong></summary>
The resulting `config.json` file would look like this.

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
        }
        {
            "name": "membrane",
            "is_particle": false,
            "label": 3
        }
    ],

    // Change this path to the location of sample_project
    "overlay_root": "local:///path/to/overlay",
    "overlay_fs_args": {
        "auto_mkdir": true
    }
}
```
</details>

## Importing Local MRC Files

**Prerequisites:** This workflow assumes:

- All tomogram files are in a flat directory structure (single folder)
- Files are in MRC format (`*.mrc`)

If you have tomograms stored locally in `*.mrc` format (e.g., from Warp, IMOD, or AreTomo), you can import them into a copick project:

```bash
copick add tomogram \
    --config config.json \
    --tomo-type denoised --voxel-size 10 --no-create-pyramid \
    'path/to/volumes.mrc'
```

### Import Parameters Explained

- `--config config.json`: Path to your copick configuration file
- `--tomo-type denoised`: Specifies the tomogram type (options: `raw`, `denoised`, `filtered`)
- `--voxel-size 10`: Sets voxel size in Ångströms (10 Å = 1 nm recommended)
- `--no-create-pyramid`: Skips pyramid generation for faster import
- `'path/to/volumes.mrc'`: Path to your MRC file(s) - supports wildcards


## Alternative Data Organizations

**Important:** If your data doesn't meet the flat directory + MRC format requirements above, please refer to our [Advanced Import Workflows](../api/import-tomos.md) documentation, which covers:

- Nested directory structures
- Different file formats (TIFF, HDF5, etc.)
- Custom import scripts
- Batch processing workflows