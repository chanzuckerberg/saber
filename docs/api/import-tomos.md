# Importing Volumes into Copick

This guide explains how to import tomograms into copick programmatically using Python. This approach is useful when you need to customize the import process, handle complex data structures, or integrate copick into existing analysis pipelines.

## Creating or Loading Runs

Copick organizes data into "runs" - individual tomographic experiments or datasets. You can either create new runs or load existing ones:

### Creating a New Run

```python
# Create a new run with a unique identifier
run = root.new_run('run001')

# You can also add metadata when creating a run
run = root.new_run('run001')
# Add custom metadata if needed
# run.meta = {'acquisition_date': '2024-01-15', 'microscope': 'Titan Krios'}
```

### Loading an Existing Run

```python
# Load an existing run by name
run = root.get_run('run001')

# List all available runs
available_runs = root.runs
print(f"Available runs: {[r.name for r in available_runs]}")
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

## Importing Tomograms

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
  

