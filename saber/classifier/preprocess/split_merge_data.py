from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import click, zarr, os
import numpy as np

def split(
    input: str,
    ratio: float,
    random_seed: int,
) -> Tuple[str, str]:
    """
    Split data from a Zarr file into training and validation sets using random split.
    Creates two new zarr files for training and validation data.
    
    Args:
        input: Path to the Zarr file
        ratio: Fraction of data to use for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing paths to:
        - Training zarr file
        - Validation zarr file
    """
    # Convert input path to Path object for easier manipulation
    input_path = Path(input)
    
    # Create output paths
    train_path = input_path.parent / f"{input_path.stem}_train.zarr"
    val_path = input_path.parent / f"{input_path.stem}_val.zarr"
    
    # Open the input Zarr file
    zfile = zarr.open(input, mode='r')
    all_keys = list(zfile.keys())
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Perform random split
    train_keys, val_keys = train_test_split(
        all_keys,
        train_size=ratio,
        random_state=random_seed
    )
    
    # Create new zarr files for training and validation
    train_zarr = zarr.open(str(train_path), mode='w')
    val_zarr = zarr.open(str(val_path), mode='w')
    
    # Copy all attributes from the input zarr file
    for attr_name, attr_value in zfile.attrs.items():
        train_zarr.attrs[attr_name] = attr_value
        val_zarr.attrs[attr_name] = attr_value
    
    # Copy data to new zarr files
    items = ['image', 'masks', 'rejected_masks']
    print('Copying data to train zarr file...')
    for key in train_keys:
        train_zarr.create_group(key)  # Explicitly create the group first
        for item in items:
            train_zarr[key][item] = zfile[key][item][:]  # [:] ensures a full copy
    
    print('Copying data to validation zarr file...')
    for key in val_keys:
        val_zarr.create_group(key)  # Explicitly create the group first
        for item in items:
            val_zarr[key][item] = zfile[key][item][:]  # [:] ensures a full copy
    
    # Print summary
    print(f"\nSplit Summary:")
    print(f"Total samples: {len(all_keys)}")
    print(f"Training samples: {len(train_keys)}")
    print(f"Validation samples: {len(val_keys)}")
    print(f"\nCreated files:")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    
    return str(train_path), str(val_path)

@click.command(context_settings={"show_default": True})
@click.option("--input", type=str, required=True, 
              help="Path to the Zarr file.")
@click.option("--ratio", type=float, required=False, default=0.8, 
              help="Fraction of data to use for training.")
@click.option("--random-seed", type=int, required=False, default=42, 
              help="Random seed for reproducibility.")
def split_data(input, ratio, random_seed):
    """
    Split data from a Zarr file into training and validation sets using random split.
    Creates two new zarr files for training and validation data.
    """

    # Call the split function
    split(input, ratio, random_seed)
    

def merge(inputs: List[str], output: str):
    """
    Merge multiple Zarr files into a single Zarr file.
    """
    # Create the output zarr group
    mergedZarr = zarr.open_group(output, mode='w')

    # Copy data from each input zarr file to the merged zarr file
    for input in inputs:
        
        # Get the session label from the input
        session_label, zarr_path = input.split(',')

        # Open the zarr file
        zfile = zarr.open_group(zarr_path, mode='r')
        keys = list(zfile.keys())

        # Copy data to new zarr files
        items = ['image', 'masks', 'rejected_masks']
        for key in keys:
            write_key = session_label + '_' + key
            mergedZarr.create_group(write_key)  # Explicitly create the group first
            for item in items:
                mergedZarr[write_key][item] = zfile[key][item][:]  # [:] ensures a full copy

    # Copy all attributes from the last input zarr file
    for attr_name, attr_value in zfile.attrs.items():
        mergedZarr.attrs[attr_name] = attr_value

    print("Merge complete!")

@click.command(context_settings={"show_default": True})
@click.option("--inputs", type=str, required=True, multiple=True,
              help="Path to the Zarr file with an associated session label provided as <session_label>,<path_to_zarr_file>.")
@click.option("--output", type=str, required=True, 
              help="Path to the output Zarr file.")
def merge_data(inputs: List[str], output: str):
    """
    Merge multiple Zarr files into a single Zarr file.
    """

    # Check if the inputs are valid
    check_inputs(inputs)

    # Merge the zarr files
    merge(inputs, output)

def check_inputs(inputs: List[str]):
    """
    Check the inputs to the merge_data command.
    """
    # Validate input format
    for input_entry in inputs:
        parts = input_entry.split(',')
        if len(parts) != 2:
            raise click.BadParameter(
                f"Invalid input format: '{input_entry}'. "
                "Each input must be in the format '<session_label>,<path_to_zarr_file>'"
            )
        session_label, zarr_path = parts
        if not session_label.strip() or not zarr_path.strip():
            raise click.BadParameter(
                f"Invalid input format: '{input_entry}'. "
                "Both session label and zarr path must be non-empty"
            )
        # Check if zarr path exists
        if not os.path.exists(zarr_path.strip()):
            raise click.BadParameter(
                f"Zarr file does not exist: '{zarr_path}'"
            )


if __name__ == '__main__':
    cli()
    

