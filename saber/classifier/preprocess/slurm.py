from saber.classifier.preprocess.micrograph_training_prep import micrograph_options
import saber.utils.slurm_submit as slurm_submit
import click

@click.command(context_settings={"show_default": True}, name='prepare-micrograph-training')
@micrograph_options
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def prepare_micrograph_training_slurm(
    input: str,
    output: str,
    target_resolution: float,
    sam2_cfg: str,
    num_gpus: int,
    gpu_constraint: str,
    ):
    """
    Prepare Training Data from Micrographs for a Classifier.
    """

    # Create Prepare Training Command
    command = f"""
saber classifier prepare-micrograph-training \\
    --input {input} \\
    --output {output} \\
    --sam2-cfg {sam2_cfg} \\
    --num-gpus {num_gpus} \\
    --gpu-constraint {gpu_constraint} \\
    """

    if target_resolution is not None:
        command += f" --target-resolution {target_resolution}"

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="prepare-micrograph-training",
        output_file="prepare-micrograph-training.out",
        shell_name="prepare-micrograph-training.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )

@click.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.sam2_inputs
@click.option('--output', type=str, required=True, help="Path to the saved SAM2 output Zarr file.", 
              default = '24jul29c_training_data.zarr')
@click.option('--num-slabs', type=int, default=1, callback=slurm_submit.validate_odd, 
              required=False, help="Number of slabs to segment per tomogram.")              
@slurm_submit.compute_commands
def prepare_tomogram_training_slurm(
    config: str,
    sam2_cfg: str,
    voxel_size: int, 
    tomogram_algorithm: str,
    slab_thickness: int,
    output: str,
    num_gpus: int,
    gpu_constraint: str,
    num_slabs: int,
    ):

    # Create Prepare Training Command
    command = f"""
saber classifier prepare-training \\
    --config {config} \\
    --sam2-cfg {sam2_cfg} \\
    --voxel-size {voxel_size} \\
    --tomo-alg {tomogram_algorithm} \\
    --slab-thickness {slab_thickness} \\
    --output {output}
    """
    
    if num_slabs > 1:
        command += f" --num-slabs {num_slabs}"

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="prepare-sam2-training",
        output_file="prepare-sam2-training.out",
        shell_name="prepare-sam2-training.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )
