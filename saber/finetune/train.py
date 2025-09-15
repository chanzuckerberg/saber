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
    deviceID: int = 0, 
    num_epochs: int = 1000):
    """
    Finetune SAM2 on tomograms and FIBs
    """

    # Determine device
    device = io.get_available_devices(deviceID)

    (cfg, checkpoint) = pretrained_weights.get_sam2_checkpoint(sam2_cfg)
    sam2_model = build_sam2(cfg, checkpoint, device=device, postprocess_mask=False)
    predictor = SAM2ImagePredictor(sam2_model)

    # Option 1 : Train the Mask Decoder and Prompt Encoder
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Load data loaders
    train_dataset = DataLoader(AutoMaskDataset(tomo_train, fib_train), batch_size=16, shuffle=True,
                               num_workers=4, pin_memory=True, collate_fn=collate_autoseg)
    val_dataset = DataLoader(AutoMaskDataset(tomo_val, fib_val), batch_size=16, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_autoseg)

    # Initialize trainer and train
    trainer = SAM2FinetuneTrainer(predictor, train_dataset, val_dataset, device)
    trainer.train(train_dataset, val_dataset, num_epochs)

    # Save Results and Model

@click.command()
@sam2_inputs
@click.option("--epochs", type=int, default=10, help="Number of epochs to train for")
@click.option("--train-zarr", type=str, help="Path to train Zarr")
@click.option("--val-zarr", type=str, help="Path to val Zarr")
def finetune(sam2_cfg: str, deviceID: int, num_epochs: int, train_zarr: str, val_zarr: str):
    """
    Finetune SAM2 on 3D Volumes. Images from input tomograms and fibs are generated with slabs and slices, respectively.
    """
    
    print("--------------------------------")
    print(
        f"Fine Tuning SAM2 on {train_zarr} and {val_zarr} for {num_epochs} epochs"
    )
    print(f"Using SAM2 Config: {sam2_cfg}")
    print(f"Using Device: {deviceID}")
    print(f"Using Number of Epochs: {num_epochs}")
    print(f"Using Train Zarr: {train_zarr}")
    print(f"Using Val Zarr: {val_zarr}")
    print("--------------------------------")

    finetune_sam2(train_zarr, val_zarr, sam2_cfg, deviceID, num_epochs)