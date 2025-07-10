# Importing Volumes into Copick with the API

This guide explains how to import tomograms into copick programmatically using Python. This approach is useful when you need to customize the import process, handle complex data structures, or integrate copick into existing analysis pipelines.

## Prerequisites

Before starting, ensure you have:
- A copick configuration file (`config.json`)
- Python environment with `copick` and `copick-utils` installed
- Your tomogram data in a supported format (NumPy arrays, MRC files, etc.)

## Setting Up Your Copick Root

First, load your copick configuration:

```python
import copick
import numpy as np

# Load your copick configuration
root = copick.from_file('config.json')
```
## Creating or Loading Runs

Copick organizes data into "runs" - individual tomographic experiments or datasets. You can either create new runs or load existing ones:

### Creating a New Run

```python
# Create a new run with a unique identifier
run = root.new_run('run001')
print(f"Created run: {run.name}")
```

### Loading an Existing Run

```python
# Load an existing run by name
run = root.get_run('run001')

# List all available runs
run_ids = [run.name for run in root.runs]
print(f"Available runs: {run_ids}")
```

## Working with Voxel Spacing

Voxel spacing defines the resolution of your tomograms. Each run can contain tomograms at multiple resolutions:

```python
# Create a new voxel spacing (10 Å recommended for most applications)
vs = run.new_voxel_spacing(10.00)

# Or load an existing voxel spacing
vs = run.get_voxel_spacing(10.00)

# List all available voxel spacings for a run
available_vs = run.voxel_spacings
print(f"Available voxel spacings: {[v.voxel_size for v in available_vs]}")
```

## Tomograms I/O

### Saving Volumes from NumPy Array

```python
# Example: Load your volume data (replace with your actual data loading)
# This could be from MRC files, TIFF stacks, HDF5, etc.
volume = np.random.rand(512, 512, 200).astype(np.float32)  # Example data

# Create a new tomogram
tomogram = vs.new_tomogram(tomo_type='denoised')

# Import the numpy array
tomogram.from_numpy(volume)
```

### Reading Volumes

Once imported, you can easily read volumes back:

```python
data = vs.get_tomogram(tomo_type='denoised').numpy()
```

## Copick-Utils

Copick-utils provides high-level functions that simplify common I/O operations:

#### Writers 

```python
from copick_utils.io import writers

# Write Tomogram
volume = np.random.rand(512, 512, 200).astype(np.float32)  # Example data
writers.tomogram(run, volume, voxel_size=10, algorithm='wbp')

# Write Segmentation
mask = np.zeros((512, 512, 200), dtype=np.uint8)
# Set some voxels to object label (e.g., 1 for ribosome)
mask[100:150, 100:150, 50:100] = 1

writers.segmentation(
    run, 
    mask, 
    user_id='user123',
    name='ribosome_segmentation', 
    session_id='0', 
    voxel_size=10
)
```

#### Readers

```python
from copick_utils.io import readers

# Read Tomogram
vol = readers.tomogram(
    run, 
    voxel_size=10, 
    algorithm='wbp', 
    raise_error=False
)

# Read Segmentation
seg = readers.segmentation(
    run,
    user_id='user123',
    name='ribosome_segmentation',
    session_id='0',
    voxel_size=10,
    raise_error=False
)
```

#### Error Handling with `raise_error`

The `raise_error` parameter controls how readers handle missing files:

- `raise_error=True` (default): Raises a `ValueError` if the file is not found
- `raise_error=False`: Returns `None` if the file is not found
