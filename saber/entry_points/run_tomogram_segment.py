import saber.utils.slurm_submit as slurm_submit
from saber.adapters.sam2.automask import amg_cli as amg
from saber import cli_context
import rich_click as click

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
    gpu_id: int = 0,
    text_prompt: str = None,
    ):
    """
    Interactive version - loads models fresh and can display results
    """

    from saber.segmenters.tomo import tomoSegmenter, multiDepthTomoSegmenter
    from saber.adapters.base import SAM2AdapterConfig, SAM3AdapterConfig
    from saber.entry_points.inference_core import segment_tomogram_core
    from saber.classifier.models import common
    import torch

    print(f"Processing {run.name} on GPU {gpu_id}")

    # Build adapter config based on whether text prompt or classifier is used
    torch.cuda.set_device(gpu_id)
    if text_prompt:
        cfg_obj = SAM3AdapterConfig(text_prompt=text_prompt)
    else:
        classifier = common.get_predictor(model_weights, model_config, gpu_id)
        cfg_obj = SAM2AdapterConfig(classifier=classifier)

    # Create segmenter based on number of slabs
    if num_slabs > 1:
        print(f'Using Multi-Depth Tomogram Segmenter with {num_slabs} slabs and {delta_z} voxel spacing between slabs.')
        segmenter = multiDepthTomoSegmenter(cfg=cfg_obj, deviceID=gpu_id, target_class=target_class)
    else:
        print(f'Using Single-Depth Tomogram Segmenter.')
        segmenter = tomoSegmenter(cfg=cfg_obj, deviceID=gpu_id)
    
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
        gpu_id=gpu_id,
        target_class=target_class,
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
    from saber.entry_points.inference_core import segment_tomogram_core
    
    # Use pre-loaded segmenter
    segmenter = models['segmenter']
    target_class = models.get('target_class', 1)

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
        gpu_id=gpu_id,
        target_class=target_class,
    )

def run_slab_seg(
    config: str,
    run_id: str,
    voxel_size: int,
    tomo_alg: str,
    slab_thickness: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str,
    npoints: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    crop_n_layers: int,
    box_nms_thresh: float,
    crop_n_points: int,
    use_m2m: bool,
    multimask: bool,
    text_prompt: str = None,
    ):
    """
    Segment a single slab of a tomogram.
    """
    from saber.segmenters.tomo import tomoSegmenter
    from saber.adapters.base import SAM2AdapterConfig
    from saber.classifier.models import common
    from copick_utils.io import readers
    from saber.adapters.sam2.amg import cfgAMG
    import copick

    # Prepare AMG Config
    cfg = cfgAMG(
        npoints = npoints, points_per_batch = points_per_batch,
        pred_iou_thresh = pred_iou_thresh, box_nms_thresh = box_nms_thresh,
        crop_n_layers = crop_n_layers, crop_n_points_downscale_factor = crop_n_points,
        use_m2m = use_m2m, multimask_output = multimask, sam2_cfg = sam2_cfg
    )

    # Open Copick Project and Get Run
    root = copick.from_file(config)
    run = root.get_run(run_id)

    # Get Tomogram
    print(f'Getting {tomo_alg} Tomogram with {voxel_size} A voxel size for the associated runID: {run.name}')
    vol = readers.tomogram(run, voxel_size, tomo_alg)
    if vol is None: # No Tomogram Found - Cancel Early
        return

    # Build adapter config based on whether text prompt or classifier is used
    if text_prompt:
        from saber.adapters.base import SAM3AdapterConfig
        adapter_cfg = SAM3AdapterConfig(text_prompt=text_prompt)
    else:
        predictor = common.get_predictor(model_weights, model_config)
        adapter_cfg = SAM2AdapterConfig(classifier=predictor, amg_cfg=cfg)

    # Create an instance of tomoSegmenter
    segmenter = tomoSegmenter(cfg=adapter_cfg)
    segmenter.save_button = True

    # For 2D segmentation, call segment_slab
    segmenter.segment_slab(
        vol, slab_thickness, 
        text=text_prompt, display=True,
        target_class=target_class
    )

def run_tomo_seg(   # run_tomograms
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
    text_prompt: str = None,
    ):
    """
    Segment a tomogram or multiple tomograms.
    """
    from saber.segmenters.loaders import tomogram_workflow
    from saber.utils import parallelization, io
    from saber.visualization import galleries 
    import copick, os, matplotlib

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
            target_class, text_prompt=text_prompt
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
        init_args=(model_weights, model_config, target_class, num_slabs),
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
        f'sID-{seg_session_id}/frames',
    )

##########################################################
# CLI Commands
##########################################################

@click.command(context_settings=cli_context, no_args_is_help=True)
@slurm_submit.copick_commands
@click.option("--run-id", "-run", type=str, required=True, 
              help="Path to Copick Config for Processing Data")            
@slurm_submit.classifier_inputs
@amg()
def slab(
    config: str,
    run_id: str,
    voxel_size: int,
    tomo_alg: str,
    slab_thickness: int,
    model_weights: str,
    model_config: str,
    target_class: int,
    sam2_cfg: str,
    npoints: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    crop_n_layers: int,
    box_nms_thresh: float,
    crop_n_points: int,
    use_m2m: bool,
    multimask: bool,
    text_prompt: str,
    ):
    """
    Segment a single slab of a tomogram.
    """

    print('🎨 Running Saber Slab Segmentation...')
    run_slab_seg(
        config, run_id, voxel_size, tomo_alg, slab_thickness, model_weights, model_config, target_class,
        sam2_cfg, npoints, points_per_batch, pred_iou_thresh, crop_n_layers, box_nms_thresh, crop_n_points,
        use_m2m, multimask, text_prompt
    )

@click.command(context_settings=cli_context, no_args_is_help=True)
@slurm_submit.copick_commands
@slurm_submit.tomogram_segment_commands
@click.option("--run-id", "-run", type=str, required=False, default=None,
              help="(Optional) RunIDs to Process. If more than one is provided, results will be displayed immediately. If None, all runs in the copick project will be processed.")
@slurm_submit.classifier_inputs
@click.option('--multi-slab', type=str, default=1, 
              help="Number of slabs and spacing for multi-slab segmentation provided as thickness or thickness,spacing. (Default spacing is 30 if ignored)")
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
    text_prompt: str,
    ):
    """
    Generate a 3D Segmentation of a tomogram.
    """

    print('🚀 Running Saber Tomogram Segmentation...')
    run_tomo_seg(
        config, run_ids, voxel_size, tomo_alg,
        seg_name, seg_session_id, slab_thickness,
        model_config, model_weights, target_class, multi_slab, text_prompt
    )
