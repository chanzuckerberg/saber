from saber.analysis.organelle_statistics import extract_organelle_statistics
from saber.utils import slurm_submit
from copick_utils.io import readers
from typing import List, Optional
import copick, click, zarr, csv
import multiprocess as mp
from tqdm import tqdm

@click.group(name="save")
@click.pass_context
def cli(ctx):
    """Save organelle coordinates and statistics (size-distribution) from segmentations."""
    pass

def common_options(func):
    """Decorator to add common options to Click commands."""
    options = [
        click.option("--config", type=str, required=True, default="path/to/copick_config.json",
                    help="Path to Copick Config for Processing Data"),
        click.option("--organelle-name", type=str, required=True,
                    help="Name of Organelle to Save in Copick"),
        click.option("--session-id", type=str, required=False, default="1",
                    help="SessionID to Save Organelle as."),
        click.option("--user-id", type=str, required=False, default="SABER",
                    help="UserID to Save Organelle as."),
        click.option("--voxel-size", type=float, required=False, default=10,
                    help="Voxel Size for the Corresponding Segmentation"),
        click.option("--run-ids", type=str, required=False, default=None,
                    help="Comma-separated list of RunIDs to process. If not provided, processes all RunIDs."),
        click.option("--n-procs", type=int, required=False, default=None,
                    help="Number of Processes to Use for Parallelization.")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def process_organelles(
    config: str,
    organelle_name: str, 
    session_id: str,
    user_id: str,
    voxel_size: float,
    run_ids: Optional[str],
    n_procs: Optional[int],
    save_copick: bool = True,
    save_statistics: bool = True,
    ):
    """Core processing function that can be used by different commands."""
    mp.set_start_method("spawn", force=True)

    # Report Input Commands
    report_input_commands(
        config, voxel_size, 
        organelle_name, session_id, 
        user_id, run_ids,
        save_copick=save_copick,
        save_statistics=save_statistics
    )

    # Read Config File
    root = copick.from_file(config)

    # Check to Ensure Pickable Object is Present in Config
    if save_copick:  pickable_object_check(root, organelle_name)

    # Parse or default to all RunIDs
    if run_ids:
        run_ids = run_ids.split(",")
        print(f"Processing the following RunIDs: {', '.join(run_ids)}\n")
    else:
        run_ids = [run.name for run in root.runs]  
    n_run_ids = len(run_ids)

    # Determine the number of processes to use
    if n_procs is None:
        n_procs = min(mp.cpu_count(), n_run_ids)
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")

    # Initialize CSV file if statistics are requested
    csv_filename = None
    if save_statistics:
        csv_filename = f'{organelle_name}_statistics.csv'
        # Create CSV with headers
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow( ['run_id', 'label', 'volume_nm3', 'diameter_nm'] )

    # Prepare arguments for each worker
    worker_args = [
        (root.get_run(run_id), organelle_name, session_id, 
         user_id, voxel_size, save_copick, save_statistics)
        for run_id in run_ids
    ]

    # Collect all CSV rows using asynchronous pool
    all_csv_rows = []
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=n_run_ids, desc="Processing", unit="run") as pbar:
            for csv_rows in pool.imap_unordered(process_single_run, worker_args):
                if csv_rows:
                    all_csv_rows.extend(csv_rows)
                pbar.update()

    # Write all CSV rows at once after processing
    if save_statistics and all_csv_rows:
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in all_csv_rows:
                writer.writerow(row)
        print(f"\nStatistics saved to {csv_filename}")

    completion_msg = []
    if save_copick:
        completion_msg.append("Coordinate extraction")
    if save_statistics:
        completion_msg.append("Statistics calculation")
    
    print(f"{' and '.join(completion_msg)} complete!")

def process_single_run(args):
    """Process a single run to extract organelle statistics and/or coordinates."""
    
    (run, organelle_name, session_id, user_id, 
     voxel_size, save_copick, save_statistics) = args
    
    # Get Segmentation Array
    seg = readers.segmentation(
        run, 
        voxel_size,
        organelle_name,
        session_id,
        user_id
    )

    # Check if Segmentation Array is Present
    if seg is None:
        print(f"{run.name} didn't have any {organelle_name} segmentations present!")
        return []

    # Extract statistics and return CSV rows
    csv_rows = extract_organelle_statistics(
        run, seg, 
        organelle_name, 
        session_id, user_id,
        voxel_size, 
        save_copick,
        save_statistics
    )

    return csv_rows if save_statistics and csv_rows else []

@cli.command(context_settings={"show_default": True})
@common_options
@click.option("--save-statistics", default=True,
              help="Save statistics to Zarr file")
def coordinates(
    config: str,
    organelle_name: str, 
    session_id: str,
    user_id: str,
    voxel_size: float,
    save_statistics: bool,
    run_ids: str,
    n_procs: int
):
    if save_statistics: description = "Coordinates and Statistics Extraction"
    else: description = "Coordinate Extraction"

    """Extract organelle coordinates and save to Copick."""
    process_organelles(
        config=config,
        organelle_name=organelle_name,
        session_id=session_id,
        user_id=user_id,
        voxel_size=voxel_size,
        run_ids=run_ids,
        n_procs=n_procs,
        save_copick=True,
        save_statistics=save_statistics
    )

@cli.command(context_settings={"show_default": True})
@common_options
def statistics(
    config: str,
    organelle_name: str, 
    session_id: str,
    user_id: str,
    voxel_size: float,
    run_ids: str,
    n_procs: int
):
    """Measure the size distribution of the measured organelles."""
    process_organelles(
        config=config,
        organelle_name=organelle_name,
        session_id=session_id,
        user_id=user_id,
        voxel_size=voxel_size,
        run_ids=run_ids,
        n_procs=n_procs,
        save_copick=False,
        save_statistics=True
    )

@cli.command(context_settings={"show_default": True})
@common_options
@click.option("--save-copick", default=False,
              help="Save coordinates to Copick")
@click.option("--save-statistics", default=True,
              help="Save statistics to Zarr file")
def slurm(
    config: str,
    organelle_name: str, 
    session_id: str,
    user_id: str,
    voxel_size: float,
    run_ids: str,
    n_procs: int,
    save_copick: bool,
    save_statistics: bool
):

    if save_copick is False and save_statistics is False:
        raise ValueError("At least one of save_copick or save_statistics must be True")

    """Submit job to SLURM for processing."""
    # Check to Ensure Pickable Object is Present in Config
    root = copick.from_file(config)
    pickable_object_check(root, organelle_name)

    # Report Input Commands
    report_input_commands(
        config, voxel_size, 
        organelle_name, session_id, 
        user_id, run_ids,
        save_copick=save_copick,
        save_statistics=save_statistics
    )
    
    # Determine which command to use
    if save_copick and save_statistics:
        command_type = "coordinates"
    elif save_copick:
        command_type = "coordinates"
    elif save_statistics:
        command_type = "statistics"
    else:
        raise ValueError("At least one of save_copick or save_statistics must be True")
    
    # Build SLURM command
    command = f"""export {command_type} \\
    --config {config} \\
    --voxel-size {voxel_size} \\
    --organelle-name {organelle_name} \\
    --session-id {session_id} --user-id {user_id} \\
    """

    if run_ids:
        command += f" --run-ids {run_ids}"
    
    if n_procs:
        command += f" --n-procs {n_procs}"

    if save_copick and save_statistics:
        command += f" --save-statistics True"

    slurm_submit.create_shellsubmit(
        job_name = f'organelle_{command_type}',
        output_file = f'organelle_{command_type}.out',
        shell_name = f'organelle_{command_type}.sh',
        command = command, 
        num_gpus = 0
    )

def report_input_commands(
    config, voxel_size, 
    organelle_name, session_id, 
    user_id, run_ids,
    save_copick=True,
    save_statistics=True
):
    """Print a summary of all inputs and processing options."""
    action_msg = []
    if save_copick:
        action_msg.append("coordinate extraction")
    if save_statistics:
        action_msg.append("statistics calculation")
    
    print(f"\nRunning organelle {' and '.join(action_msg)} with the following parameters:\n"
          f"\tConfig: {config}\n"
          f"\tOrganelle Name: {organelle_name}\n"
          f"\tSession ID: {session_id}\n"
          f"\tUser ID: {user_id}\n"
          f"\tVoxel Size: {voxel_size}\n"
          f"\tRun IDs: {run_ids if run_ids else 'All'}\n"
          f"\tSave to Copick: {save_copick}\n"
          f"\tSave Statistics: {save_statistics}\n")

def pickable_object_check(root, organelle_name):
    """Check if the specified organelle exists as a pickable object in the config."""
    objects = root.pickable_objects
    if not any(obj.name == organelle_name for obj in objects):
        # Print all available object names
        available_names = [obj.name for obj in objects]
        available_names = f"Available pickable object names: {', '.join(available_names)}"
        raise ValueError(f"Pickable Object {organelle_name} not found in Config!\n{available_names}")
