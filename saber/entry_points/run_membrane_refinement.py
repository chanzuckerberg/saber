from saber.analysis.refine_membranes import OrganelleMembraneFilter
from copick_utils.io import readers, writers
from typing import List, Tuple, Optional
from saber.utils import parallelization
import click, copick

@click.group()
@click.pass_context
def cli(ctx):
    pass

def convert_info(ctx, param, value):
    if not value:
        return None
    
    parts = value.split(',')
    if len(parts) == 1:
        # Only name provided
        return (parts[0].strip(), None, None)
    elif len(parts) == 3:
        # name, userID, sessionID provided
        return (parts[0].strip(), parts[1].strip(), parts[2].strip())
    else:
        raise click.BadParameter(f"Invalid format '{value}'. Expected 'name' or 'name,userID,sessionID'")

def refine_membranes_options(func):
    """Decorator to add shared options for refine_membranes commands."""
    options = [
        click.option("--config", type=str, required=False, default='config.json',
                      help="Path to the config file."),
        click.option("--org-info", type=str, required=False, default='organelles,saber,1', callback=convert_info,
                      help="Path to the organelle info file. Provide either name or name, userID, sessionID as a comma separated string."),
        click.option("--mem-info", type=str, required=False, default='membranes,membrane-seg,1', callback=convert_info,
                      help="Path to the membrane info file. Provide either name or name, userID, sessionID as a comma separated string."),
        click.option("--voxel-size", type=float, required=False, default=10,
                      help="Voxel Size of the Segmentation [Angstroms]."),
        click.option('--save-session-id', type=str, required=False, default='1',
                      help="Session ID to save the refined segmentations to."),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve order in CLI
        func = option(func)
    return func

@cli.command(context_settings={"show_default": True})
@refine_membranes_options
def refine_membranes(
    config: str,
    voxel_size: float,
    org_info: Tuple[str, Optional[str], Optional[str]],
    mem_info: Tuple[str, Optional[str], Optional[str]],
    save_session_id: str,
    ):
    """Refine organelle and membrane segmentations using morphological filtering."""

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]
    
    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=refine_membranes_workflow,
        approach="threading",
        verbose=True
    )

    # Prepare tasks
    tasks = [
        (run, org_info, mem_info, voxel_size, save_session_id)
        for run in root.runs ]

    # Execute
    try:
        pool.execute(
            run_refinement,
            tasks, task_ids=run_ids,
            progress_desc="Refining Membranes"
        )     
    finally:
        pool.shutdown()

    # Report Results to User
    print('Completed the Membrane Refinement!')        

def run_refinement(run, org_info, mem_info, voxel_size, save_session_id, gpu_id, models):

    refiner = models

    # Get the Segmentations
    org_seg = readers.segmentation(run, voxel_size, org_info[0], session_id=org_info[2], user_id=org_info[1])
    mem_seg = readers.segmentation(run, voxel_size, mem_info[0], session_id=mem_info[2], user_id=mem_info[1])

    # Return None if one of the segmentations is not found
    if org_seg is None:
        print(f'No Organele Segmentation Found for {run.name}')
        return
    elif mem_seg is None:
        print(f'No Membrane Segmentation Found for {run.name}')
        return

    # Run the Refinement
    results = refiner.run(org_seg,mem_seg)

    # Convert the Results to 3D Labels
    mem_seg = refiner.convert_to_3d_labels(results['membranes'])
    org_seg = refiner.convert_to_3d_labels(results['organelles'])

    # Save the results
    mem_user_id = return_write_user_id(mem_info[1], run)
    writers.segmentation(run, mem_seg, mem_user_id, name=mem_info[0], session_id=save_session_id, voxel_size=voxel_size)
    
    org_user_id = return_write_user_id(org_info[1], run)
    writers.segmentation(run, org_seg, org_user_id, name=org_info[0], session_id=save_session_id, voxel_size=voxel_size)

    return results

def return_write_user_id(user_id, run):
    if user_id is None:
        return 'saber-refined'
    else:
        return user_id + '-refined'

def refine_membranes_workflow(gpu_id:int):
    return OrganelleMembraneFilter(gpu_id = gpu_id) 

if __name__ == '__main__':
    cli()   