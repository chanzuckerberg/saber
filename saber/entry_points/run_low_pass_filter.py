from saber.filters.tomograms import Filter3D
from copick_utils.io import writers, readers
from saber.utils import io, slurm_submit
import click, mrcfile, os, glob, json
from tqdm import tqdm

@click.group(name="filter3d")
@click.pass_context
def cli(ctx):
    """Apply low-pass and high-pass filters to tomograms."""
    pass

def low_pass_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--lp-freq", type=float, required=False, default=0, 
                    help="Low-pass cutoff frequency (in Angstroms)"),
        click.option("--lp-decay", type=float, required=False, default=0, 
                    help="Low-pass decay width (in pixels)"),
        click.option("--hp-freq", type=float, required=False, default=0, 
                    help="High-pass cutoff frequency (in Angstroms)"),
        click.option("--hp-decay", type=float, required=False, default=0, 
                    help="High-pass decay width (in pixels)"),
        click.option("--show-filter", type=bool, required=False, default=False, 
                    help="Save the filter as a Png (filter3d.png)")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def copick_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--config", type=str, required=True, 
                    help="Path to Copick Config for Processing Data"),
        click.option("--run-ids", type=str, required=False, default=None,  
                    help="Run ID to process (No Input would process the entire dataset.)"),    
        click.option("--tomo-alg", type=str, required=True, 
                    help="Tomogram Algorithm to use"),
        click.option("--voxel-size", type=float, required=False, default=10, 
                    help="Voxel Size to Query the Data"),   
        click.option("--show-filter", type=bool, required=False, default=True, 
                    help="Save the filter as a Png (filter3d.png)")                                 
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def mrc_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--read-path", type=str, required=True, 
                    help="Path to MRC File to Process"),
        click.option("--save-path", type=str, required=True, 
                    help="Path to Save the Processed MRC File"),
        click.option("--voxel-size", type=float, required=True, default=10, 
                    help="Voxel Size of the Tomograms")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def input_check(lp_freq, hp_freq, voxel_size):
    if lp_freq == 0 and hp_freq == 0:
        raise ValueError("Low-pass and high-pass frequencies cannot both be 0.")
    elif lp_freq < voxel_size * 2:
        raise ValueError("Low-pass frequency cannot be less than twice the Nyquist resolution.")
    elif hp_freq < voxel_size * 2:
        raise ValueError("High-pass frequency cannot be less than twice the Nyquist resolution.")
    elif lp_freq > hp_freq and lp_freq > 0 and hp_freq > 0:
        raise ValueError("Low-pass cutoff resolution must be less than high-pass cutoff resolution.")        

def print_header(lp_freq, lp_decay, hp_freq, hp_decay):
    print('----------------------------------------')
    print(f'Low-Pass Frequency: {lp_freq} Angstroms')
    print(f'Low-Pass Decay: {lp_decay} Pixels')
    print(f'High-Pass Frequency: {hp_freq} Angstroms')
    print(f'High-Pass Decay: {hp_decay} Pixels')
    print('----------------------------------------')

@cli.command(context_settings={"show_default": True})
@low_pass_commands
@copick_commands
def copick(
    config: str,
    run_ids: str,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    tomo_alg: str,
    voxel_size: float,
    show_filter: bool 
    ):
    import copick

    input_check(lp_freq, hp_freq, voxel_size)

    # Load Copick Project
    if os.path.exists(config):  root = copick.from_file(config)
    else:                       raise ValueError(f"Config file {config} does not exist.")

    print_header(lp_freq, lp_decay, hp_freq, hp_decay)
    
    # Get Run IDs
    if run_ids is None: run_ids = [run.name for run in root.runs]
    else:               run_ids = run_ids.split(",")

    # Determine Write Algorithm
    write_algorithm = tomo_alg
    if lp_freq > 0: write_algorithm = write_algorithm + f'-lp{lp_freq:0.0f}A'
    if hp_freq > 0: write_algorithm = write_algorithm + f'-hp{hp_freq:0.0f}A'

    # Get Tomogram for Initializing 3D Filter
    vol = readers.tomogram(root.get_run(run_ids[0]), voxel_size, tomo_alg)

    # Create 3D Filter
    filter = Filter3D(
        apix=voxel_size, sz=vol.shape, 
        lp= lp_freq, lpd = lp_decay, 
        hp=hp_freq, hpd=hp_decay)

    # Save Filter
    if show_filter:
        filter.show_filter()   
    
    # Get Tomogram and Process
    for run_id in tqdm(run_ids):

        run = root.get_run(run_id)
        vol = readers.tomogram(run, voxel_size, tomo_alg)

        # Apply Low-pass Filter
        vol = filter.apply(vol)

        # Write Tomogram
        writers.tomogram(run, vol.cpu().numpy(), voxel_size, write_algorithm)

    print('Applying Filters to All Tomograms Complete...')

@cli.command(context_settings={"show_default": True})
@low_pass_commands
@copick_commands
def copick_slurm(  
    config: str,
    run_ids: str,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    tomo_alg: str,
    voxel_size: float,
    show_filter: bool):

    input_check(lp_freq, hp_freq, voxel_size)
    
    command = f"""filter3d copick \\
    --config {config} \\
    --tomo-alg {tomo_alg} \\
    --voxel-size {voxel_size} \\
    """

    # Add Low-Pass Filter and High-Pass Filter
    if lp_freq > 0:
        command += f" --lp-freq {lp_freq} --lp-decay {lp_decay}"
    if hp_freq > 0:
        command += f" --hp-freq {hp_freq} --hp-decay {hp_decay}"
        
    if run_ids is not None:
        command += f" --run-ids {run_ids}"

    # Add Show Filter if not desired
    if not show_filter:
        command += f" --show-filter {show_filter}"

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="filter3d-copick",
        output_file="filter3d-copick.out",
        shell_name="filter3d.sh",
        command=command,
        num_gpus=1,
        gpu_constraint="a100"
    )    

###################################################################################################

@cli.command(context_settings={"show_default": True})
@low_pass_commands
@mrc_commands
def mrc(
    read_path: str,
    save_path: str,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    voxel_size: float,
    show_filter: bool = False
    ):

    input_check(lp_freq, hp_freq, voxel_size)

    # Get Tomogram Paths
    run_ids = glob.glob(os.path.join(read_path, '*.mrc'))

    # Get Tomogram for Initializing 3D Filter
    vol = mrcfile.open(run_ids[0], mode='r').data

    # Create 3D Filter
    filter = Filter3D(
        apix=voxel_size, sz=vol.shape, 
        lp= lp_freq, lpd = lp_decay, 
        hp=hp_freq, hpd=hp_decay)

    # Save Filter
    if show_filter:
        filter.show_filter()       

    os.makedirs(save_path, exist_ok=True)
    write_parameters(read_path, save_path, voxel_size, lp_freq, lp_decay, hp_freq, hp_decay)
    
    # Get Tomogram and Process
    for run_id in tqdm(run_ids):

        # Get Tomogram
        vol = mrcfile.open(run_id, mode='r').data

        # Apply Low-pass Filter
        vol = filter.apply(vol)

        # Write Tomogram
        with mrcfile.new(os.path.join(save_path, run_id.split('/')[-1]), overwrite=True) as mrc:
            mrc.set_data(vol.cpu().numpy())
            mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
            mrc.update_header_stats()


def write_parameters(read_path, save_path, voxel_size, lp_freq, lp_decay, hp_freq, hp_decay):
    parameters = {
        "Read Path": read_path,
        "Save Path": save_path,
        "Voxel Size": voxel_size,
        "Low-Pass Frequency (Angstroms)": lp_freq,
        "Low-Pass Decay (Pixels)": lp_decay,
        "High-Pass Frequency (Angstroms)": hp_freq,
        "High-Pass Decay (Pixels)": hp_decay
    }
    # Print the parameters dictionary
    print(json.dumps(parameters, indent=4))
    
    with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, indent=4)
