import saber.utils.slurm_submit as slurm_submit
import click

@click.group()
@click.pass_context
def cli(ctx):
    """
    SLURM Submission for Segmentation
    """

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.tomogram_segment_commands
@click.option("--run-ids", type=str, required=False, default=None, 
              help="Path to Copick Config for Processing Data")
@slurm_submit.compute_commands
@slurm_submit.classifier_inputs 
@slurm_submit.sam2_inputs
def tomograms_slurm(
    config: str,
    run_ids: str,
    voxel_size: float, 
    tomogram_algorithm: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    num_gpus: int,
    gpu_constraint: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str
    ):
    """
    Generate a SLURM submission to segment a tomogram.
    """

    command = f"""
saber segment tomograms \\
    --config {config} \\
    --slab-thickness {slab_thickness} \\
    --voxel-size {voxel_size} --tomo-alg {tomogram_algorithm} \\
    --segmentation-name {segmentation_name} --segmentation-session-id {segmentation_session_id} \\
    """

    if  model_config is not None:
        command += f""" --model-config {model_config} --target-class {target_class}"""
    3
    if run_ids is not None:
        command += f""" --run-ids {run_ids}"""

    # Create Slurm Submit Script
    slurm_submit.create_shellsubmit(
        job_name="sam2-segment",
        output_file="sam2-segment.out",
        shell_name="sam2-segment.sh",
        command=command,
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint
    )

@cli.command(context_settings={"show_default": True})
@slurm_submit.classifier_inputs
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def micrographs_slurm(
    input: str,
    output: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
    target_resolution: float,
    num_gpus: int,
    gpu_constraint: str
    ):
    """
    Generate a SLURM submission to segment all micrographs in a project.
    """
        
    pass

@cli.command(context_settings={"show_default": True})
def refine_membranes_slurm(
    config: str,
    org_info: str,
    mem_info: str,
    ):
    pass


if __name__ == '__main__':
    cli()