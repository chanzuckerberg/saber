from __future__ import annotations

from saber import cli_context
from typing import List, Tuple
import rich_click as click

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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sklearn.model_selection import train_test_split
    from saber.utils.progress import _progress 
    from rich.console import Console
    from threading import Lock
    from pathlib import Path
    import numpy as np
    import zarr, os

    # Initialize Rich console for output
    console = Console()

    # Convert input path to Path object for easier manipulation
    input_path = Path(input)
    
    # Create output paths
    train_path = input_path.parent / f"{input_path.stem}_train.zarr"
    val_path = input_path.parent / f"{input_path.stem}_val.zarr"
    
    # Open the input Zarr file
    zfile = zarr.open_group(input, mode='r')
    all_keys = list(zfile.keys())
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Perform random split
    train_keys, val_keys = train_test_split(
        all_keys,
        train_size=ratio,
        random_state=random_seed
    )

    console.rule("[bold]Split Summary")
    console.print(f"[b]Source:[/b] {input}")
    console.print(f"Total samples: {len(all_keys)}")
    console.print(f"Training samples: {len(train_keys)}")
    console.print(f"Validation samples: {len(val_keys)}\n")
    
    # Create new zarr files for training and validation
    train_zarr = zarr.open(str(train_path), mode='w')
    val_zarr = zarr.open(str(val_path), mode='w')
    
    # Copy all attributes from the input zarr file
    for attr_name, attr_value in zfile.attrs.items():
        train_zarr.attrs[attr_name] = attr_value
        val_zarr.attrs[attr_name] = attr_value
    
    # Define items to copy
    items = ['0', 'labels/0', 'labels/rejected']
    
    # Function to copy a single key
    def copy_key(key, source_zarr, dest_zarr):
        """Copy a single key from source to destination zarr."""
        dest_zarr.create_group(key)
        copy_attributes(source_zarr[key], dest_zarr[key])
        for item in items:
            try:
                dest_zarr[key][item] = source_zarr[key][item][:]
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to copy {key}/{item}: {e}[/yellow]")
        copy_attributes(source_zarr[key]['labels'], dest_zarr[key]['labels'])
    
    # Parallel copy for training data
    max_workers = num_workers = min(os.cpu_count() or 8, len(train_keys))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key in train_keys:
            future = executor.submit(copy_key, key, zfile, train_zarr)
            futures.append(future)
        
        # Wait for completion with progress bar
        for _ in _progress(as_completed(futures), total=len(futures), description="Copying train data"):
            pass
    
    # Parallel copy for validation data
    max_workers = num_workers = min(os.cpu_count() or 8, len(val_keys))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key in val_keys:
            future = executor.submit(copy_key, key, zfile, val_zarr)
            futures.append(future)
        
        # Wait for completion with progress bar
        for _ in _progress(as_completed(futures), total=len(futures), description="Copying validation data"):
            pass
    
    # Print summary
    console.rule("[bold]Created files")
    console.print(f"[b]Training data:[/b] {train_path}")
    console.print(f"[b]Validation data:[/b] {val_path}\n")
    
    return str(train_path), str(val_path)


def merge(inputs: List[str], output: str):
    """
    Merge multiple Zarr files into a single Zarr file.

    Args:
        inputs: List of input Zarr files
        output: Path to the output Zarr file
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from saber.utils.progress import _progress 
    from rich.console import Console
    from threading import Lock
    import zarr, os 

    console = Console()
    attr_lock = Lock()  # Lock for thread-safe attribute updates

    console.rule("[bold cyan]Zarr Merge")
    console.print(f"[b green]Creating merged zarr file at:[/b green] {output}")
    
    # Create the output zarr group
    mergedZarr = zarr.open_group(output, mode='w')
    
    # Define items to copy
    items = ['0', 'labels/0', 'labels/rejected']
    
    # Function to copy a single key
    def copy_key(key, session_label, source_zarr, dest_zarr):
        """Copy a single key from source to destination zarr."""
        write_key = f"{session_label}_{key}"
        
        # Create the group and copy its attributes
        new_group = dest_zarr.create_group(write_key)
        copy_attributes(source_zarr[key], new_group)
        
        # Copy the data arrays
        for item in items:
            try:
                dest_zarr[write_key][item] = source_zarr[key][item][:]
            except Exception as e:
                pass  # Silently skip missing items
        
        # Copy attributes for labels subgroup
        copy_attributes(source_zarr[key]['labels'], new_group['labels'])
    
    # Process each input zarr file
    for input_spec in _progress(inputs, description="Merging inputs"):
        # Get the session label from the input
        session_label, zarr_path = input_spec.split(',')
        
        # Open the zarr file
        zfile = zarr.open_group(zarr_path, mode='r')
        keys = list(zfile.keys())
        
        # Parallel copy of keys
        max_workers = min(os.cpu_count() or 8, len(keys))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key in keys:
                future = executor.submit(copy_key, key, session_label, zfile, mergedZarr)
                futures.append(future)
            
            # Wait for completion with progress bar
            for _ in _progress(as_completed(futures), total=len(futures), 
                             description=f"[{session_label}] merging keys"):
                pass
        
        # Copy all attributes from the input zarr file (thread-safe)
        with attr_lock:
            for attr_name, attr_value in zfile.attrs.items():
                if attr_name not in mergedZarr.attrs:
                    mergedZarr.attrs[attr_name] = attr_value

    console.rule("[bold green]Merge complete!")
    console.print(f"[b white]Output file:[/b white] {output}")


def check_inputs(inputs: List[str]):
    """
    Check the inputs to the merge_data command.
    """
    import os

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


def copy_attributes(source, destination):
    """
    Copy all attributes from source zarr object to destination zarr object.
    
    Args:
        source: Source zarr group/array with attributes to copy
        destination: Destination zarr group/array to copy attributes to
    """
    if hasattr(source, 'attrs') and source.attrs:
        destination.attrs.update(source.attrs)


########################################################
# Merge Data Command
########################################################

@click.command(context_settings=cli_context)
@click.option("-i", "--inputs", type=str, required=True, multiple=True,
              help="Path to the Zarr file with an associated session label provided as <session_label>,<path_to_zarr_file>.")
@click.option("-o", "--output", type=str, required=False, default='labeled.zarr',
              help="Path to the output Zarr file.")
def merge_data(inputs: List[str], output: str):
    """
    Merge multiple Zarr files into a single Zarr file.

    Example:
        saber classifier merge-data --inputs session1,/path/to/session1.zarr --inputs session2,/path/to/session2.zarr --output merged.zarr
    """

    # Check if the inputs are valid
    check_inputs(inputs)

    # Merge the zarr files
    merge(inputs, output)
    

########################################################
# Split Data Command
########################################################

@click.command(context_settings=cli_context)
@click.option("-i", "--input", type=str, required=True, 
              help="Path to the Zarr file.")
@click.option('-r', "--ratio", type=float, required=False, default=0.8, 
              help="Fraction of data to use for training.")
@click.option("--random-seed", type=int, required=False, default=42, 
              help="Random seed for reproducibility.")
def split_data(input, ratio, random_seed):
    """
    Split data from a Zarr file into training and validation sets using random split.
    Creates two new zarr files for training and validation data.

    Example:
        saber classifier split-data --i data.zarr --ratio 0.8 --max-workers 16
    """

    # Call the split function
    split(input, ratio, random_seed)