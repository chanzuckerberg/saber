from saber.classifier.datasets import singleZarrDataset, multiZarrDataset, augment
from saber.classifier.trainer import ClassifierTrainer
from saber.classifier.models import common
from saber.utils import io, slurm_submit

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch, click, yaml, os, zarr, json
from torch.utils.data import DataLoader
from monai.losses import FocalLoss
from monai.transforms import Compose
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn

@click.group()
@click.pass_context
def cli(ctx):
    pass

def run(
    train_path: str,
    validate_path: str,
    num_epochs: int,
    batch_size: int,
    num_classes: int,
    backbone: str,
    model_size: str,
    model_weights: str,
    ):

    # Set device
    device = io.get_available_devices()

    # Initialize model
    model = common.get_classifier_model(backbone, num_classes, model_size)
    
    # Load model weights if Fine-Tuning
    if model_weights:
        model.load_state_dict(torch.load(model_weights, weights_only=True))

        # # Freeze all parameters except classifier
        #     for name, param in model.named_parameters():
        #         if 'classifier' not in name:
        #             param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    else: # Start with Higher Learning Rate if not Fine-Tuning
        optimizer = AdamW(model.parameters(), lr=5e-4)
    model = model.to(device)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Create datasets and dataloaders
    print('Loading training data...')
    train_loader, train_dataset = get_dataloaders(train_path, 'train', batch_size)
    print('Loading validation data...')
    val_loader, _ = get_dataloaders(validate_path, 'val', batch_size)
    
    # Option 1: Initialize CrossEntropyLoss
    # print('Estimating class weights...')
    # label_counts = {i: 0 for i in range(num_classes)}
    # for sample in tqdm(train_dataset):
    #     label_counts[sample['label'].item()] += 1
    # # Compute class weights using inverse frequency normalization
    # total_samples = sum(label_counts.values())
    # class_weights = [
    #     total_samples / (len(label_counts) * label_counts[i]) if label_counts[i] > 0 else 0
    #     for i in range(len(label_counts))
    # ]
    # class_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # print(f'Class weights: {class_weights}')
    # loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    
    # Option 2: Initialize MONAI's FocalLoss
    loss_fn = FocalLoss(gamma=1, alpha=0.5, reduction="mean")

    # # Initialize trainer and Train
    print('Training...')
    trainer = ClassifierTrainer(model, optimizer, scheduler, loss_fn, device)
    trainer.results_path = f'results_{backbone}_{model_size}'
    trainer.train(train_loader, val_loader, num_epochs)

    # # Save results to Zarr
    trainer.save_results(train_path, validate_path)

    # Save Model Config
    model_config = {
        'model': {
            'backbone': backbone,
            'model_size': model_size,
            'num_classes': num_classes,
            'weights': os.path.abspath(os.path.join(trainer.results_path, 'best_model.pth'))
        },
        'optimizer': {
            'optimizer': optimizer.__class__.__name__,
            'scheduler': scheduler.__class__.__name__,
            'loss_fn': loss_fn.__class__.__name__
        },
        'data': {
            'train': train_path,
            'validate': validate_path,
            'classes': get_class_names(train_path)
        }
    }

    with open(f'results_{backbone}_{model_size}/model_config.yaml', 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False, sort_keys=False, indent=2)

def get_dataloaders(zarr_path: str, mode: str, batch_size: int):

    # Select appropriate transforms
    if mode == 'train':
            transforms = Compose([augment.get_preprocessing_transforms(True), 
                                  augment.get_training_transforms()])
    else:   transforms = Compose([augment.get_validation_transforms()])

    # Load dataset
    # Check if the string contains commas, indicating multiple paths
    if ',' in zarr_path:
        # Split by comma and create a list of paths
        path_list = [path.strip() for path in zarr_path.split(',')]
        dataset = multiZarrDataset.MultiZarrDataset(path_list, mode=mode, transform=transforms)
    else:
        # Single path
        dataset = singleZarrDataset.ZarrSegmentationDataset(zarr_path, mode=mode, transform=transforms)
    print(f'Dataset length: {len(dataset)}')
    
    # Create dataloader - Only Shuffle for training
    if mode == 'train': loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:               loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return loader, dataset

def train_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--train", type=str, required=True, 
                    help="Path to the Zarr(s) file. In the format 'file.zarr' or 'file1.zarr,file2.zarr'."),
        click.option("--validate", type=str, required=False, default=None,
                    help="Path to the Zarr(s) file. In the format 'file.zarr' or 'file1.zarr,file2.zarr'."),
        click.option("--num-epochs", type=int, default=10, 
                    help="Number of epochs to train for."),
        click.option("--num-classes", type=click.IntRange(min=2), default=2, 
                    help="Number of classes to train for - background + Nclasses\n(2 is binary classification)."),
        click.option("--backbone", default="SAM2",
                    type=click.Choice(['ConvNeXt', 'SwinTransformer', 'SAM2', 'cryoDinoV2'], case_sensitive=False),
                    help="Backbone to use for training."),
        click.option("--model-size", default="large",
                    type=click.Choice(['tiny', 'small', 'base', 'large'], case_sensitive=False),
                    help="Model size to use for training."),
        click.option("--batch-size", type=int, default=32, 
                    help="Batch size for training."),
        click.option("--model-weights", type=str, default=None,
                    help="Path to the model weights to use for training.")
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func

@click.command(context_settings={"show_default": True})
@train_commands
def train(
    train: str,
    validate: str,
    num_epochs: int,
    num_classes: int,
    batch_size: int,
    backbone: str,
    model_size: str,
    model_weights: str,
    ):
    """
    Train a Classifier.
    """

    run(train, validate, num_epochs, batch_size, num_classes, backbone, model_size, model_weights)

@click.command(context_settings={"show_default": True})
@train_commands
def train_slurm(
    train: str,
    validate: str,
    num_epochs: int,
    num_classes: int,
    batch_size: int,
    backbone: str,
    model_size: str,
    model_weights: str,
    ):
    """
    Train a Classifier.
    """

    # Use triple quotes for the multi-line f-string
    command = f"""classifier train \\
        --train {train} \\
        --validate {validate} \\
        --num-epochs {num_epochs} \\
        --num-classes {num_classes} \\
        --batch-size {batch_size} \\
        --backbone {backbone} \\
        --model-size {model_size}"""

    if model_weights:
        command += f" --model-weights {model_weights}"

    # Create a slurm job
    slurm_submit.create_shellsubmit(
        job_name = "train_classifier",
        output_file = "train_classifier.out",
        shell_name = "train_classifier.sh",
        command = command
    )


def get_class_names(zarr_path: str):
    """
    Get the class names from the Zarr file.
    The class names are stored as a string in the Zarr file.
    This function converts the string to a dictionary.
    """

    # Open the Zarr file
    zfile = zarr.open(zarr_path, mode='r')

    # Get the class names
    class_names = zfile.attrs['class_names']
    
    # convert to dict
    return json.loads(class_names) 