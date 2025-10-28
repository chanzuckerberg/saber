from saber.segmenters.tomo import cryoTomoSegmenter, multiDepthTomoSegmenter
from saber.entry_points.inference_core import segment_tomogram_core
from saber.segmenters.loaders import tomogram_workflow
import saber.utils.slurm_submit as slurm_submit
import copick, click, torch, os, matplotlib
from saber.utils import parallelization, io
from saber.classifier.models import common
from saber.visualization import galleries 
from copick_utils.io import readers
import os, yaml

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
    if vol is None: # No Tomogram Found - Cancel Early
        return

    # Create an instance of cryoTomoSegmenter
    segmenter = cryoTomoSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,         # if you have a classifier; otherwise, leave as None
        target_class=target_class     # desired target class if using a classifier
    )
    segmenter.save_button = True

    # For 2D segmentation, call segment_image
    segmenter.segment_slab(vol, slab_thickness, display_image=True)

@cli.command(context_settings={"show_default": True})
@slurm_submit.copick_commands
@slurm_submit.tomogram_segment_commands
@click.option("--run-ids", type=str, required=False, default=None,
              help="(Optional) RunIDs to Process. If more than one is provided, results will be displayed immediately. If None, all runs in the copick project will be processed.")
@slurm_submit.classifier_inputs
@click.option('--multi-slab', type=str, default=1, 
              help="Number of slabs and spacing for multi-slab segmentation provided as thickness or thickness,spacing. (Default spacing is 30 if ignored)")
@slurm_submit.sam2_inputs
def tomograms(
    config: str,
    run_ids: str,
    voxel_size: float, 
    tomo_alg: str,
    seg_name: str,
    seg_session_id: str,
    slab_thickness: int,
    model_config: str,
    model_weights: str,
    target_class: int,
    multi_slab: str,
    sam2_cfg: str
    ):
    """
    Generate a 3D Segmentation of a tomogram.
    """

    print(f'\nRunning SAM2 Organelle Segmentations for the Following Tomograms:\n Algorithm: {tomo_alg}, Voxel-Size: {voxel_size} Å')

    # Parse multi-slab option
    delta_z = 30  # Default Spacing between slabs
    if ',' in multi_slab:
        num_slabs, delta_z = map(int, multi_slab.split(','))
    else:
        num_slabs = int(multi_slab)

    # Open Copick Project and Query All Available Runs
    root = copick.from_file(config)

    # Get RunIDs from Copick Project
    display_segmentation = False
    if run_ids is None:
        run_ids = [run.name for run in root.runs]
    elif ',' in run_ids:
        run_ids = run_ids.split(',')
    else:
        run = root.get_run(run_ids)
        display_segmentation = True
        segment_tomogram_interactive(
            run, voxel_size, tomo_alg,
            seg_name, seg_session_id,
            slab_thickness, num_slabs, delta_z,
            display_segmentation,
            model_weights, model_config,
            target_class, sam2_cfg
        )
        return

    # # Set to Agg Backend to Avoid Displaying Matplotlib Figures
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg')

    # Save Metadata
    metadict = {
        'inputs': {
            'config': config,
            'voxel_size': voxel_size,
            'tomo_alg': tomo_alg,
            'slab_thickness': slab_thickness,
            'multi_slab': multi_slab,
        },
        'model': {
            'config': model_config,
            'weights': model_weights,
            'target_class': target_class,
        },
        'outputs': {
            'name': seg_name,
            'sessionID': seg_session_id,
            'userID': 'saber',
        } }
    io.save_copick_metadata(config, metadict, f'segment-saber_{seg_session_id}_{seg_name}.yaml')

    # Create pool with model pre-loading
    pool = parallelization.GPUPool(
        init_fn=tomogram_workflow,
        approach="threading",
        init_args=(model_weights, model_config, target_class, sam2_cfg, num_slabs),
        verbose=True
    )

    # Prepare tasks (same format as your existing code)
    tasks = [
        (root.get_run(id), voxel_size, tomo_alg, seg_name,
         seg_session_id, slab_thickness, num_slabs, delta_z,
         display_segmentation)
        for id in run_ids
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
        f'{seg_name}_gallery/frames',
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
    delta_z: int,
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

    # Create segmenter based on number of slabs
    if num_slabs > 1:
        print(f'Using Multi-Depth Tomogram Segmenter with {num_slabs} slabs and {delta_z} voxel spacing between slabs.')
        segmenter = multiDepthTomoSegmenter(
            sam2_cfg=sam2_cfg,
            deviceID=gpu_id,
            classifier=classifier,
            target_class=target_class
        )
    else:
        print(f'Using Single-Depth Tomogram Segmenter.')
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
        delta_z=delta_z,
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
    delta_z: int,
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
        num_slabs=num_slabs, delta_z=delta_z,
        display_segmentation=display_segmentation,
        segmenter=segmenter,
        gpu_id=gpu_id
    )