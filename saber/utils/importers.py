from saber.filters.downsample import FourierRescale3D
import copick, mrcfile, glob, os, click
from copick_utils.io import writers
from tqdm import tqdm

@click.group(name="preprocess")
@click.pass_context
def cli(ctx):
    """Import tomograms from local MRC/MRCS files."""
    pass

def run_from_mrcs(
    mrcs_path,
    config,
    target_tomo_type,
    input_voxel_size,
    output_voxel_size = 10):
    """
    Import and process tomograms from local MRC/MRCS files.
    
    Args:
        mrcs_path (str): Path to directory containing MRC/MRCS files
        config (str): Path to the copick configuration file
        target_tomo_type (str): Name to use for the tomogram locally
        input_voxel_size (float): Original voxel size of the tomograms
        output_voxel_size (float, optional): Desired voxel size for downsampling
    """
    # Load Copick Project
    if os.path.exists(config):
        root = copick.from_file(config)
    else:
        raise ValueError('Config file not found')

    # List all .mrc and .mrcs files in the directory
    mrc_files = glob.glob(os.path.join(mrcs_path, "*.mrc")) + glob.glob(os.path.join(mrcs_path, "*.mrcs"))
    if not mrc_files:
        print(f"No .mrc or .mrcs files found in {mrcs_path}")
        return

    # Prepare rescaler if needed
    rescale = None
    if input_voxel_size is not None and output_voxel_size > input_voxel_size:
        rescale = FourierRescale3D(input_voxel_size, output_voxel_size)        

    # Check if the mrcs file exists
    if not os.path.exists(mrcs_path):
        raise FileNotFoundError(f'MRCs file not found: {mrcs_path}')
    
    for mrc_path in tqdm(mrc_files):

        # Get or Create Run
        runID = os.path.splitext(os.path.basename(mrc_path))[0]
        try:
            run = root.new_run(runID)
        except Exception as e:
            run = root.get_run(runID)

        # Load the mrcs file
        with mrcfile.open(mrc_path) as mrc:
            vol = mrc.data
            # Check voxel size in MRC header vs user input
            mrc_voxel_size = float(mrc.voxel_size.x)  # assuming cubic voxels
            if abs(mrc_voxel_size - input_voxel_size) > 1e-1:
                print(f"WARNING: Voxel size in {mrc_path} header ({mrc_voxel_size}) "
                      f"differs from user input ({input_voxel_size})")

        # Rescale if needed
        if rescale is not None:
            vol = rescale.run(vol)
            voxel_size_to_write = output_voxel_size
        else:
            voxel_size_to_write = input_voxel_size

        # Write the tomogram
        writers.tomogram(run, vol, voxel_size_to_write, target_tomo_type)
    print(f"Processed {len(mrc_files)} files from {mrcs_path}")

@cli.command(context_settings={"show_default": True})
@click.option('--mrcs-path', type=str, required=True, help='Path to the mrcs file')
@click.option('--config', type=str, required=True, help='Path to the config file to write tomograms to')
@click.option('--target-tomo-alg', type=str, required=True, help='Reconstruction algorithm used to create the tomogram')
@click.option('--input-voxel-size', type=float, required=False, default=None, help='Voxel size of the tomogram')
@click.option('--output-voxel-size', type=float, required=False, default=10, help='Save voxel size')
def from_mrcs(mrcs_path, config, target_tomo_alg, input_voxel_size, output_voxel_size):
    """
    Command-line interface for importing MRC/MRCS files.
    Handles argument parsing and calls from_mrcs with the parsed arguments.
    """
    run_from_mrcs(mrcs_path, config, target_tomo_alg, input_voxel_size, output_voxel_size)

def run_empty_picks(
    config,
    particle_name, 
    out_user,
    out_session,
    overwrite = False
):
    """
    Create empty picks for a given particle name.

    Taken from https://github.com/copick/copick-catalog/blob/main/solutions/copick/create_empty_picks/solution.py
    """
    # Load Copick Project
    if os.path.exists(config):
        root = copick.from_file(config)
    else:
        raise ValueError('Config file not found')

    # Create picks
    for run in tqdm(root.runs):

        picks = run.get_picks(
            object_name=particle_name, user_id=out_user, session_id=out_session
        )

        if len(picks) == 0:
            picks = run.new_picks(
                object_name=particle_name, user_id=out_user, session_id=out_session
            )
        else:
            if overwrite:
                picks = picks[0]
            else:
                raise ValueError(
                    f"Picks already exist for {run.name}. Set overwrite to True to overwrite."
                )

        picks.points = []
        picks.store()

@cli.command(context_settings={"show_default": True})
@click.option('--config', type=str, required=True, help='Path to the config file to write tomograms to')
@click.option('--particle-name', type=str, required=True, help='Name of the particle to create picks for')
@click.option('--out-user', type=str, required=False, default='copick', help='User ID to write picks to')
@click.option('--out-session', type=str, required=False, default='0', help='Session ID to write picks to')
@click.option('--overwrite', type=bool, required=False, default=False, help='Overwrite existing picks')
def empty_picks(
    config,
    particle_name,
    out_user,
    out_session,
    overwrite = False
):
    run_empty_picks(config, particle_name, out_user, out_session, overwrite)