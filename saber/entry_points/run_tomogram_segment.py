from saber.entry_points.inference_core import segment_tomogram_core
from saber.segmenters.loaders import tomogram_workflow
from saber.segmenters.tomo import cryoTomoSegmenter
import saber.utils.slurm_submit as slurm_submit
import copick, click, torch, os, matplotlib
from saber.classifier.models import common
from saber.visualization import galleries 
from saber.utils import parallelization
from copick_utils.io import readers

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@click.option("--run-id", type=str, required=True, 
              help="Path to Copick Config for Processing Data")            
@slurm_submit.classifier_inputs
@slurm_submit.sam2_inputs
def slab(
    config: str,
    run_id: str, 
    voxel_size: int, 
    tomo_alg: str,
    slab_thickness: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str
    ):
    """
    Segment a single slab of a tomogram.
    """

    # Initialize the Domain Expert Classifier   
    predictor = common.get_predictor(model_weights, model_config)

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)

    # Get Run
    run = root.get_run(run_id)

    # Get Tomogram
    print(f'Getting {tomo_alg} Tomogram with {voxel_size} A voxel size for the associated runID: {run.name}')
    vol = readers.tomogram(run, voxel_size, tomo_alg)

    # Create an instance of cryoTomoSegmenter
    segmenter = cryoTomoSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,         # if you have a classifier; otherwise, leave as None
        target_class=target_class     # desired target class if using a classifier
    )

    # For 2D segmentation, call segment_image
    masks = segmenter.segment_slab(vol, slab_thickness, display_image=True)

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.tomogram_segment_commands
@click.option("--run-ids", type=str, required=False, default=None,
              help="Path to Copick Config for Processing Data")
@slurm_submit.classifier_inputs
@click.option("--num-slabs", type=int, default=1, callback=slurm_submit.validate_odd,
              required=False, help="Number of Slabs to Segment")
@slurm_submit.sam2_inputs
def tomograms(
    config: str,
    run_ids: str,
    voxel_size: float, 
    tomo_alg: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    model_config: str,
    model_weights: str,
    target_class: int,
    num_slabs: int,
    sam2_cfg: str
    ):
    """
    Generate a 3D Segmentation of a tomogram.
    """

    print(f'\nRunning SAM2 Organelle Segmentations for the Following Tomograms:\n Algorithm: {tomo_alg}, Voxel-Size: {voxel_size} Å')

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)

    # Get RunIDs from Copick Project
    if run_ids is None:
        display_segmentation = False
        run_ids = [run.name for run in root.runs]
    else:
        run = root.get_run(run_ids)
        display_segmentation = True   
        segment_tomogram_interactive(
            run, voxel_size, tomo_alg,
            segmentation_name, segmentation_session_id,
            slab_thickness, num_slabs,
            display_segmentation,
            model_weights, model_config,
            target_class, sam2_cfg
        )
        return

    # # Set to Agg Backend to Avoid Displaying Matplotlib Figures
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg')

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=tomogram_workflow,
        approach="threading",
        init_args=(model_weights, model_config, target_class, sam2_cfg),
        verbose=True
    )

    # Prepare tasks (same format as your existing code)
    tasks = [
        (run, voxel_size, tomo_alg, segmentation_name,
         segmentation_session_id, slab_thickness, num_slabs, 
         display_segmentation)
        for run in root.runs
    ]

    # Execute
    try:
        pool.execute(
            segment_tomogram_parallel,
            tasks, task_ids=run_ids,
            progress_desc="Segmenting Tomograms"
        )
            
    finally:
        pool.shutdown()
    
    # Report Results to User
    print('Completed the Orgnalle Segmentations with Cryo-SAM2!')

    # Create a gallery of the tomograms
    galleries.create_png_gallery(
        f'gallery_sessionID_{segmentation_session_id}/frames',
    )

# Segment a Single Tomogram
def segment_tomogram_interactive(
    run,
    voxel_size: float,
    tomo_alg: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    num_slabs: int,
    display_segmentation: bool,
    model_weights: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str,
    gpu_id: int = 0
    ):
    """
    Interactive version - loads models fresh and can display results
    """
    
    print(f"Processing {run.name} on GPU {gpu_id}")
    
    # Load models fresh for interactive use
    torch.cuda.set_device(gpu_id)
    classifier = common.get_predictor(model_weights, model_config, gpu_id)
    segmenter = cryoTomoSegmenter(
        sam2_cfg=sam2_cfg,
        deviceID=gpu_id,
        classifier=classifier,
        target_class=target_class
    )
    
    # Call core function
    segment_tomogram_core(
        run=run,
        voxel_size=voxel_size,
        tomogram_algorithm=tomo_alg,
        segmentation_name=segmentation_name,
        segmentation_session_id=segmentation_session_id,
        slab_thickness=slab_thickness,
        num_slabs=num_slabs,
        display_segmentation=display_segmentation,
        segmenter=segmenter,
        gpu_id=gpu_id
    )
    
# Segment Tomograms with GPUPool
def segment_tomogram_parallel(
    run,
    voxel_size: float,
    tomo_alg: str,
    segmentation_name: str,
    segmentation_session_id: str,
    slab_thickness: int,
    num_slabs: int,
    display_segmentation: bool,
    gpu_id,     # Added by GPUPool
    models      # Added by GPUPool
    ):
    """
    Parallel version - uses pre-loaded models from GPUPool
    """
    
    # Use pre-loaded segmenter
    segmenter = models['segmenter']
    
    # Call core function
    segment_tomogram_core(
        run=run,
        voxel_size=voxel_size,
        tomogram_algorithm=tomo_alg,
        segmentation_name=segmentation_name,
        segmentation_session_id=segmentation_session_id,
        slab_thickness=slab_thickness,
        num_slabs=num_slabs,
        display_segmentation=display_segmentation,
        segmenter=segmenter,
        gpu_id=gpu_id
    )
