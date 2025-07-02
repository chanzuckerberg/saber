import json, click, saber

@click.group()
@click.pass_context
def cli(ctx):
    pass

def validate_num_gpus(ctx, param, value):
    if value is not None and (value < 1 or value > 4):
        raise click.BadParameter("Number of GPUs must be between 1 and 4.")
    return value

def create_shellsubmit(
    job_name, 
    output_file,
    shell_name,
    command,
    num_gpus = 1, 
    gpu_constraint = 'h100'):

    if num_gpus > 0:
        slurm_gpus = f'#SBATCH --nodes=1\n#SBATCH --partition=gpu\n#SBATCH --gpus={gpu_constraint}:{num_gpus}'
    else:
        slurm_gpus = f'#SBATCH --partition=cpu'

    shell_script_content = f"""#!/bin/bash

{slurm_gpus}
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}

ml anaconda 
conda activate /hpc/projects/group.czii/conda_environments/pySAM2

{command}
"""
    with open(shell_name, 'w') as file:
        file.write(shell_script_content)

    print(f"\nShell script {shell_name} created successfully.\n")


################################### COMMON CLI COMMANDS ##############################################


def copick_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
    click.option("--config", type=str, required=True, default="path/to/copick_config.json",
                 help="Path to Copick Config for Processing Data"),
    click.option("--voxel-size", type=float, required=False, default=10, 
                 help="Resolution of Desired Tomograms to Process"),
    click.option("--tomo-alg", type=str, required=False, default='denoised', 
                 help="Reconstrution Algorithm to Query Tomgorams"),    
    click.option("--slab-thickness", type=float, required=False, default=10, 
                 help="Thickness of Slab for Producing Initial Segmentation"),                              
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def tomogram_segment_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
    click.option("--segmentation-name", type=str, required=False, default="organelles", 
                 help="Name of Segmentation Session"),
    click.option("--segmentation-session-id", type=str, required=False, default="1", 
                 help="SessionID to Write for Segmentation Mask"),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def compute_commands(func):
    """Decorator to add common slurm submission compute options to a Click command."""
    options = [
        click.option("--num-gpus", type=int, required=False, default=2, callback=validate_num_gpus, 
                     help="Number of GPUs for Processing"),
        click.option("--gpu-constraint", required=False, default="h100", 
                     type=click.Choice(['a6000', 'h200', 'h100', 'a100'], case_sensitive=False), 
                     help="GPU to Select for Processing)")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def classifier_inputs(func):
    """Decorator to add common options for the Classifier command."""
    options = [
        click.option("--model-config", type=str,required=False, default=None,
                     help="Path to Classifier Model Config"),
        click.option("--model-weights", type=str, required=False, default=None,
                    help="Path to Classifier model trained weights."),    
        click.option("--target-class", type=int, required=False, default=-1,
                    help="Target Class for Segmentation. When set to -1, the model performs semantic segmentation.\nWhen set to a positive integer, the model performs instance segmentation for the desired class..")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

def sam2_inputs(func):
    """Decorator to add common options for the SAM2 command."""
    options = [
        click.option("--sam2-cfg", required=False, default='large', help="SAM2 Model Config",
                     type=click.Choice(['large', 'base', 'small', 'tiny'], case_sensitive=False))
        # click.option("--min-mask-area", type=int, required=False, default=100,
        #              help="Minimum Area of Mask to Keep"),
        # click.option("--min-rel-box-size", type=float, required=False, default=0.01,
        #              help="Minimum Relative Box Size of Mask to Keep")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


# Callback to ensure the --multiple-slabs parameter is an odd integer (and at least 1)
def validate_odd(ctx, param, value):
    if value < 0:
        raise click.BadParameter("The --multiple-slabs parameter must be at least 1.")
    if value % 2 == 0:
        raise click.BadParameter("The --multiple-slabs parameter must be an odd number.")
    return value

if __name__ == "__main__":
    cli()
