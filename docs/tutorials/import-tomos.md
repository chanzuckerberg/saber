# Importing Volumes

SABER uses [Copick](https://github.com/copick/copick) to manage tomographic data. Copick provides a unified interface for accessing volumes whether they're stored locally, on an HPC cluster, or on the [CryoET Data Portal](https://cryoetdataportal.czscience.com).

---

## Setting Up a Copick Project

=== "New Project (Local Files)"

    Create a config pointing to a local directory where results will be written:

    ```bash
    copick config filesystem \
        --overlay-root /path/to/overlay
    ```

    To define biological objects at creation time (useful for particle picking workflows):

    ```bash
    copick config filesystem \
        --overlay-root /path/to/overlay \
        --objects ribosome,True,130,6QZP \
        --objects apoferritin,True,65 \
        --objects membrane,False
    ```

    ??? example "What the generated config.json looks like"

        ```json
        {
            "name": "my_project",
            "description": "A test project.",
            "version": "1.0.0",

            "pickable_objects": [
                { "name": "ribosome",   "is_particle": true,  "label": 1, "radius": 130, "pdb_id": "6QZP" },
                { "name": "apoferritin","is_particle": true,  "label": 2, "radius": 65 },
                { "name": "membrane",   "is_particle": false, "label": 3 }
            ],

            "overlay_root": "local:///path/to/overlay",
            "overlay_fs_args": { "auto_mkdir": true },

            "static_root": "local:///path/to/static",
            "static_fs_args": { "auto_mkdir": true }
        }
        ```

        !!! note "Overlay vs. Static root"
            The **overlay root** is writable — SABER writes segmentations, coordinates, and metadata here. The **static root** is read-only, intended for the original tomogram files that you never want to accidentally overwrite.

=== "CryoET Data Portal"

    Link a project to a public dataset from the [CryoET Data Portal](https://cryoetdataportal.czscience.com):

    ```bash
    copick config dataportal \
        --dataset-id DATASET_ID \
        --overlay-root /path/to/local/overlay
    ```

    Pickable objects are automatically populated from the portal dataset. You only need to provide the dataset ID and a local path for writing results.

    !!! tip "Finding your dataset ID"
        Browse datasets at [cryoetdataportal.czscience.com](https://cryoetdataportal.czscience.com) and copy the numeric ID from the dataset URL.

---

## Importing Local MRC Files

If you have tomograms processed with Warp, IMOD, AreTomo, or any other reconstruction pipeline, import them into Copick with:

```bash
copick add tomogram \
    --config config.json \
    --tomo-type denoised \
    --voxel-size 10 \
    --no-create-pyramid \
    'path/to/volumes/*.mrc'
```

| Parameter | Description |
|-----------|-------------|
| `--tomo-type` | Tag for this reconstruction (`denoised`, `wbp`, `filtered`) |
| `--voxel-size` | Voxel size in Å — SABER uses this for downsampling |
| `--no-create-pyramid` | Skip multi-resolution pyramid (faster import) |

!!! warning "Flat directory required"
    This command expects all `.mrc` files in a single flat directory. If your files are in nested subdirectories, see the [Advanced Import](../api/import-tomos.md) guide.

---

## Verifying Your Project

After import, confirm that Copick can see your tomograms:

```python
import copick

root = copick.from_file("config.json")
print(f"Found {len(root.runs)} runs")

# Inspect the first run
run = root.runs[0]
print(run.name)
print([t.voxel_size for t in run.tomograms])
```

---

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-arrow-right-24: **Pre-processing**](preprocessing.md)

    Generate SAM2 segmentations and annotate them in the GUI.

-   [:octicons-arrow-right-24: **Advanced Import**](../api/import-tomos.md)

    Non-MRC formats, nested directories, and custom import scripts.

-   [:octicons-arrow-right-24: **Copick Documentation**](https://copick.github.io/copick/)

    Full Copick reference including remote filesystem setup.

</div>
