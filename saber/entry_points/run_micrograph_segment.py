import saber.process.slurm_submit as slurm_submit
from saber.microSABER import cryoMicroSegmenter
import click, torch, os
from tqdm import tqdm
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

def segment_micrograph_separate_process(
    mrc_path: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int
):

    pass

@cli.command(context_settings={"show_default": True})
@slurm_submit.classifier_inputs
@click.option("--target-class", type=int, required=False, default=1,
              help="Target Class for Segmentation")
def image(
    mrc_path: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int
    ):
    """
    Segment a single micrograph.
    """
    pass

@cli.command(context_settings={"show_default": True})
@slurm_submit.classifier_inputs
@click.option("--target-class", type=int, required=False, default=1,
              help="Target Class for Segmentation")
def mrc_project(
    project_path: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int
    ):
    """
    Segment all micrographs in a project.
    """
    import multiprocess as mp
    mp.set_start_method("spawn")
    
    pass 
    
@cli.command(context_settings={"show_default": True})
@slurm_submit.classifier_inputs
@click.option("--target-class", type=int, required=False, default=1,
              help="Target Class for Segmentation")
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def mrc_project_slurm(
    project_path: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    num_gpus: int,
    gpu_constraint: str
    ):
    """
    Generate a SLURM submission to segment all micrographs in a project.
    """
    
    pass