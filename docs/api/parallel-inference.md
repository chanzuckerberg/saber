# Asyncronous Multi-GPU Inference

The GPUPool class is a powerful tool in SABER that enables parallel inference across multiple GPUs. This class efficiently distributes deep learning models across available GPU devices, allowing for asynchronous and parallel processing of large datasets.

## Overview
GPUPool manages the distribution of tasks across available GPUs and handles the initialization of models on each device. It provides a streamlined interface for:

* Initializing models on multiple GPUs
* Distributing tasks across available GPU resources
* Managing asynchronous execution with proper error handling
* Tracking progress of batch processing operations

## Usage

As a high overview, here is an complete example of applying parallel inference of SABER on 2D data with `GPUPool`.

```python
from saber.entry_points import parallelization
import glob

# Get all micrograph files
files = glob.glob("path/to/micrographs/*.mrc")

# Create processing pool
pool = parallelization.GPUPool(
    init_fn=initialize_model,
    init_args=("large", model_weights, model_config, target_class),
    verbose=True
)

# Prepare tasks
tasks = [(fname, "output.zarr", 2) for fname in files]

# Execute batch processing
try:
    pool.execute(
        process_task,
        tasks, 
        task_ids=files,
        progress_desc="Processing micrographs"
    )
finally:
    pool.shutdown()
```

The general workflow involves 3 key steps:

### 1. **Define a task list:**

```python
tasks = [(input_file1, output_path1, params1), 
         (input_file2, output_path2, params2),
         ...]
```

### 2. **Create Model Initialization Function**
Define a function that initializes your model on a specific GPU. The GPU ID must be the first parameter:

```python
from saber.segmenters.micro import cryoMicroSegmenter
from saber.classifier.models import common

def initialize_model(
    gpu_id:int, 
    model_weights:str, model_config:str, 
    target_class:int, sam2_cfg:str):
    """Load micrograph segmentation models once per GPU"""
    
    torch.cuda.set_device(gpu_id)
    
    # Load models
    predictor = common.get_predictor(model_weights, model_config, gpu_id)
    segmenter = cryoMicroSegmenter(
        sam2_cfg=sam2_cfg,
        deviceID=gpu_id,
        classifier=predictor,
        target_class=target_class
    )
    
    return {
        'predictor': predictor,
        'segmenter': segmenter
    }
```
### 3. **Create Processing Function**
Define a function that processes each task using the initialized model. The GPU ID and model must be the last two parameters:

```python
def process_task(
    input:str, output: str,
    scale_factor: float, gpu_id, models):

    # Get the Global Zarr Writer
    zwriter = zarr_writer.get_zarr_writer(output)

    # Use pre-loaded segmenter
    segmenter = models['segmenter']        

    # Ensure we're on the correct GPU
    torch.cuda.set_device(gpu_id)
    
    # Read the Micrograph
    image, pixel_size = io.read_micrograph(input)
    image = image.astype(np.float32)

    # Downsample the input image
    image = FourierRescale2D.run(image, scale_factor)   

    # Produce Initialial Segmentations with SAM2
    segmenter.segment( image, display_image=False )
    (image0, masks_list) = (segmenter.image0, segmenter.masks)

    # Convert Masks to Numpy Array
    masks = mask_filters.masks_to_array(masks_list)

    # Write Run to Zarr
    input = os.path.splitext(os.path.basename(input))[0]
    zwriter.write(run_name=input, image=image0, masks=masks.astype(np.uint8))
```

## Key Parameters

### Initialization
```python
GPUPool(
    init_fn,                # Function to initialize model on each GPU
    init_args=None,         # Arguments passed to init_fn
    init_kwargs=None,       # Keyword arguments passed to init_fn
    gpu_ids=None,           # Specific GPU IDs to use (defaults to all available)
    num_workers=None,       # Number of worker processes (defaults to number of GPUs)
    verbose=False           # Enable/disable verbose output
)
```

### Execution
```python
execute(
    fn,                    # Function to execute on each task
    tasks,                 # List of task parameters
    task_ids=None,         # Optional identifiers for each task
    progress_desc=None,    # Description for progress bar
    **kwargs               # Additional keyword arguments passed to fn
)
```
