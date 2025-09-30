from saber.classifier.datasets.augment import get_finetune_transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor
from saber.finetune.trainer import SAM2FinetuneTrainer
from saber.finetune.dataset import AutoMaskDataset
from saber.finetune.helper import collate_autoseg
from saber.utils.slurm_submit import sam2_inputs
from torch.utils.data import DataLoader
from sam2.build_sam import build_sam2
from saber import pretrained_weights
from saber.utils import io
import click

def finetune_sam2(
    tomo_train: str = None, 
    fib_train: str = None, 
    tomo_val: str = None, 
    fib_val: str = None, 
    sam2_cfg: str = 'base', 
    num_epochs: int = 1000,
    batch_size: int = 16):
    """
    Finetune SAM2 on tomograms and FIBs
    """

    # Determine device
    (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
    sam2_model = build_sam2(cfg, checkpoint, device='cuda', postprocess_mask=False)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Option 1 : Train the Mask Decoder and Prompt Encoder
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Load data loaders
    train_loader = DataLoader( AutoMaskDataset(
                               tomo_train, fib_train, transform=get_finetune_transforms(), 
                               slabs_per_volume_per_epoch=20 ),
                               batch_size=batch_size, shuffle=True, 
                               num_workers=4, pin_memory=True, collate_fn=collate_autoseg 
                            )

    val_loader = DataLoader( AutoMaskDataset(
                             tomo_val, fib_val, slabs_per_volume_per_epoch=15 ),
                             num_workers=4, pin_memory=True, collate_fn=collate_autoseg,
                             batch_size=batch_size, shuffle=False ) if (tomo_val or fib_val) else train_loader

    # Initialize trainer and train
    trainer = SAM2FinetuneTrainer( predictor, train_loader, val_loader )
    # trainer.train( num_epochs, best_metric='AR' )
    trainer.train( num_epochs )

@click.command()
@sam2_inputs
@click.option("--fib-train", type=str, help="Path to train Zarr")
@click.option("--fib-val", type=str, help="Path to val Zarr")
@click.option("--tomo-train", type=str, help="Path to train Zarr")
@click.option("--tomo-val", type=str, help="Path to val Zarr")
@click.option("--epochs", type=int, default=1000, help="Number of epochs to train for")
@click.option('--batch-size', type=int, default=16, help="Batch size")
def finetune(sam2_cfg: str, epochs: int, fib_train: str, fib_val: str, tomo_train: str, tomo_val: str, batch_size: int):
    """
    Finetune SAM2 on 3D Volumes. Images from input tomograms and fibs are generated with slabs and slices, respectively.
    """
    
    print("--------------------------------")
    print(
        f"Fine Tuning SAM2 on {fib_train} and {fib_val} and {tomo_train} and {tomo_val} for {epochs} epochs"
    )
    print(f"Using SAM2 Config: {sam2_cfg}")
    print(f"Tomo Train Zarr: {tomo_train}")
    print(f"Tomo Val Zarr: {tomo_val}")
    print(f"Fib Train Zarr: {fib_train}")
    print(f"Fib Val Zarr: {fib_val}")
    print(f"Using Number of Epochs: {epochs}")
    print(f"Using Batch Size: {batch_size}")
    print("--------------------------------")

    finetune_sam2(tomo_train, fib_train, tomo_val, fib_val, sam2_cfg, epochs, batch_size)