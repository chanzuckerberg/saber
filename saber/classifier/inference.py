from saber.visualization import galleries, classifier as classviz
from saber.classifier.models.predictor import Predictor
from saber.filters import masks as mask_filters
from saber.utils import slurm_submit
import zarr, click, torch, os
from tqdm import tqdm
import numpy as np

def predict_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--model-weights", type=str, required=True, 
                    help="Path to the model weights file."),
        click.option("--model-config", type=str, required=True, 
                    help="Path to the model config file."),
        click.option("--input", type=str, required=True, 
                    help="Path to the Zarr file."),
        click.option("--output", type=str, required=True, 
                    help="Path to the output Zarr file."),                 
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

@click.command(context_settings={"show_default": True})
@predict_commands
def predict(model_weights, model_config, input, output):
    """
    Run inference on all images/masks in the Zarr file and store results in a new Zarr file.
    """

    # Load model    
    if os.path.exists(model_weights) and os.path.exists(model_config):
        predictor = Predictor(model_config, model_weights)
        num_classes = predictor.config['model']['num_classes']
    else:
        raise FileNotFoundError(f"Model config file {model_config} or model weights file {model_weights} does not exist.")

    # Load Zarr file
    if os.path.exists(input):
        zfile = zarr.open(input, mode='r')
        run_ids = list(zfile.keys())
    else:
        raise FileNotFoundError(f"Zarr file {input} does not exist.")

    # Get Image Dimensions
    (nx,ny) = zfile[run_ids[0]]['image'].shape

    # Create an output Zarr store
    output_zfile = zarr.open(output, mode='w')

    # Create a label image for all masks
    final_masks = np.zeros([num_classes - 1, nx, ny], dtype=np.uint8)
    
    # Main Loop
    for run_id in tqdm(run_ids):

        im = np.array(zfile[run_id]['image'])
        masks = np.array(zfile[run_id]['masks'])

        # Run batched inference directly
        # predictions = predictor.predict(im, masks)  # Shape: (Nmasks, num_classes)
        predictions = predictor.batch_predict(im, masks)  # Shape: (Nmasks, num_classes)

        # Initialize the final masks
        final_masks[:] = 0

        # Apply the classifier to the masks
        for class_idx in range(1, num_classes):
            mask_predictions = mask_filters.convert_predictions_to_masks(predictions, masks, class_idx)
            
            # If there are any masks, sum them up
            if len(mask_predictions) > 0:
                
                all_masks = [mask['segmentation'] for mask in mask_predictions]
                final_masks[class_idx-1] = (np.sum(all_masks, axis=0) > 0).astype(np.uint8)
        
        # Option: Display the final masks
        # classviz.display_masks(im, final_masks)

        # Save the combined class masks and corresponding image 
        output_zfile.create_dataset(f"{run_id}/masks", data=final_masks)
        output_zfile.create_dataset(f"{run_id}/image", data=im)

        # Clear variables to free memory
        del im, masks, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert Zarr to PNG Galleries 
    galleries.convert_zarr_to_gallery(output)

    print(f"Inference completed. Results saved to {output}")

@click.command(context_settings={"show_default": True})
@predict_commands
def predict_slurm(
    model_weights, 
    model_config, 
    input, 
    output):

    command = f"""classifier predict \\
    --model-weights {model_weights} \\
    --model-config {model_config} \\
    --input {input} \\
    --output {output}"""
    
    slurm_submit.create_shellsubmit(
        job_name = "predict_classifier",
        output_file = "predict_classifier.out",
        shell_name = "predict_classifier.sh",
        command = command
    )