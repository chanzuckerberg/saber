from saber.process.downsample import FourierRescale2D
import saber.process.slurm_submit as slurm_submit
from saber.microSABER import cryoMicroSegmenter
import mrcfile, skimage, glob, click, torch
from saber.classifier.models import common
from saber.io import read_micrograph
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Try to import hyperspy for Material Science dataset
try:
    import hyperspy.api as hs
except:
    pass

@click.group()
@click.pass_context
def cli(ctx):
    pass

def segment_micrograph_separate_process(
    input: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    display_image: bool = False,
    use_sliding_window: bool = True,
    target_resolution: float = None
    ):

    # Initialize the Domain Expert Classifier   
    predictor = common.get_predictor(model_weights, model_config)

    # Create an instance of cryoMicroSegmenter
    segmenter = cryoMicroSegmenter(
        sam2_cfg=sam2_cfg,
        classifier=predictor,          # if you have a classifier; otherwise, leave as None
        target_class=target_class     # desired target class if using a classifier
    )

    # Get Micrograph
    print(f'Getting Micrograph from {input}')
    im, pixel_size = read_micrograph(input)
    im = im.astype(np.float32)

    # Downsample if desired resolution is larger than current resolution
    if target_resolution is not None and target_resolution > pixel_size:
        scale = target_resolution / pixel_size
        im = FourierRescale2D.run(im, scale)

#     # Temporary Crop
#     im = im[75:75+1100, 1280:1280+1100]
#     im -= 30000
#     im[im < 0] = 0
# #    plt.imshow(im,cmap='gray'); plt.show()
# #    import pdb; pdb.set_trace()

    # Segment Micrograph
    masks = segmenter.segment_image(
        im,
        display_image=display_image,
        use_sliding_window=use_sliding_window,
        )

@cli.command(context_settings={"show_default": True})
@click.option("--input", type=str, required=True,
              help="Path to Micrograph or Project, in the case of project provide the file extention (e.g. path/*.mrc)")
@slurm_submit.sam2_inputs
@slurm_submit.classifier_inputs
@click.option("--sliding-window", type=bool, required=False, default=False,
              help="Use Sliding Window for Segmentation")
@click.option("--target-resolution", type=float, required=False, default=None, 
              help="Desired Resolution to Segment Images [Angstroms]. If not provided, no downsampling will be performed.")
def micrographs(
    input: str,
    sam2_cfg: str,
    model_weights: str,
    model_config: str,
    target_class: int,
    sliding_window: bool,
    target_resolution: float 
    ):
    """
    Segment a single micrograph or all micrographs in a folder.
    """

    
    files = glob.glob(input)
    if len(files) == 0:
        raise ValueError(f"No files found in {input}")
    elif len(files) == 1:
        segment_micrograph_separate_process(
            files[0], sam2_cfg, model_weights, model_config, target_class,
            display_image=True, use_sliding_window=sliding_window, 
            target_resolution=target_resolution,
            )
    else:
        import multiprocess as mp
        mp.set_start_method("spawn")

        n_procs = torch.cuda.device_count()
        print(f'\nRunning SAM2 Organelle Segmentations for {len(files)} Micrographs:\n')
        print(f'Paraellizing the Computation over {n_procs} GPUs\n')

        # Main Loop - Segment All Micrographs
        for _iz in range(0, len(files), n_procs):
            processes = []
            for _in in range(n_procs):
                _iz_this = _iz + _in
                if _iz_this >= len(files):
                    break
                file = files[_iz_this]
                print(f'\nProcessing {file} ({iter}/{len(files)})')    
                p = mp.Process(
                    target=segment_micrograph_separate_process,
                    args=(file, 
                        sam2_cfg,
                        model_weights,
                        model_config,
                        target_class,
                        False,
                        sliding_window,
                        ),
                )
                processes.append(p)
                iter += 1

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            for p in processes:
                p.close()
    
@cli.command(context_settings={"show_default": True})
@slurm_submit.classifier_inputs
@click.option("--target-class", type=int, required=False, default=1,
              help="Target Class for Segmentation")
@slurm_submit.sam2_inputs
@slurm_submit.compute_commands
def micrographs_slurm(
    input: str,
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

