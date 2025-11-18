import rich_click as click

@click.group()
@click.pass_context
def cli(ctx):
    pass

def run(
    train_path: str,
    validate_path: str,
    num_epochs: int,
    batch_size: int,
    model_weights: str,
    ):
    from saber.classifier.fabric_trainer import ClassifierTrainerFabrics
    from saber.classifier.trainer import ClassifierTrainer
    from saber.classifier.models import common

    from torch.optim.lr_scheduler import CosineAnnealingLR
    from monai.losses import FocalLoss
    from torch.optim import AdamW
    from saber.utils import io
    import torch, yaml, os

    # Get the Model Size from the Train Zarr File and Initialize 
    (labels, amg_params) = get_metadata(train_path)
    num_classes = len(labels)

    # Initialize trainer and Train
    ngpus = torch.cuda.device_count()
    device_id = -1 if ngpus > 1 else 0
    model = common.get_classifier_model(
            'SAM2', num_classes,
            amg_params['sam2_cfg'], 
            deviceID=device_id
        )
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # Create datasets and dataloaders
    print('Loading training data...')
    train_loader, train_dataset = get_dataloaders(train_path, 'train', batch_size)
    print('Loading validation data...')
    val_loader, _ = get_dataloaders(validate_path, 'val', batch_size)
    
    # Option 2: Initialize MONAI's FocalLoss
    loss_fn = FocalLoss(gamma=2, alpha=0.5, reduction="mean")

    # Start Training
    if ngpus == 1:
        print('Training...')
        # Set device
        device = io.get_available_devices()
        model = model.to(device)
        trainer = ClassifierTrainer(model, optimizer, scheduler, loss_fn, device)
    else:
        trainer = ClassifierTrainerFabrics(model, optimizer, scheduler, loss_fn)

    trainer.results_path = 'results'
    trainer.train(train_loader, val_loader, num_epochs)

    # Save results to Zarr
    trainer.save_results(train_path, validate_path)

    # Save Model Config
    model_config = {
        'model': {
            'num_classes': num_classes,
            'weights': os.path.abspath(os.path.join(trainer.results_path, 'best_model.pth')),
        },
        'labels': labels,
        'data': {
            'train': train_path,
            'validate': validate_path
        },
        'amg_params': amg_params,
        'optimizer': {
            'optimizer': optimizer.__class__.__name__,
            'scheduler': scheduler.__class__.__name__,
            'loss_fn': loss_fn.__class__.__name__, 
            'num_epochs': num_epochs,
            'batch_size': batch_size
        },
    }

    with open(f'results/model_config.yaml', 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False, sort_keys=False, indent=2)

def get_dataloaders(zarr_path: str, mode: str, batch_size: int):
    from saber.classifier.datasets import singleZarrDataset, multiZarrDataset, augment
    from torch.utils.data import DataLoader
    from monai.transforms import Compose

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

#################################### CLI Commands ####################################

def train_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option('-t', "--train", type=str, required=True, 
                    help="Path to the Zarr(s) file. In the format 'file.zarr' or 'file1.zarr,file2.zarr'."),
        click.option("-v", "--validate", type=str, required=False, default=None,
                    help="Path to the Zarr(s) file. In the format 'file.zarr' or 'file1.zarr,file2.zarr'."),
        click.option("-ne", "--num-epochs", type=int, default=10, 
                    help="Number of epochs to train for."),
        click.option("-bs", "--batch-size", type=int, default=32, 
                    help="Batch size for training."),
        click.option("-mw", "--model-weights", type=str, default=None,
                    help="Model weights used for fine-tuning.")
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
    batch_size: int,
    model_weights: str,
    ):
    """
    Train a Classifier.
    """
    run(train, validate, num_epochs, batch_size, model_weights)

@click.command(context_settings={"show_default": True})
@train_commands
def train_slurm(
    train: str,
    validate: str,
    num_epochs: int,
    batch_size: int,
    model_weights: str,
    ):
    from saber.utils import slurm_submit
    """
    Train a Classifier.
    """

    # Use triple quotes for the multi-line f-string
    command = f"""classifier train \\
        --train {train} \\
        --validate {validate} \\
        --num-epochs {num_epochs} \\
        --batch-size {batch_size} """

    if model_weights:
        command += f" --model-weights {model_weights}"

    # Create a slurm job
    slurm_submit.create_shellsubmit(
        job_name = "train_classifier",
        output_file = "train_classifier.out",
        shell_name = "train_classifier.sh",
        command = command
    )


def get_metadata(zarr_path: str):
    import zarr
    """
    Get the class names from the Zarr file.
    The class names are stored as a string in the Zarr file.
    This function converts the string to a dictionary.
    """

    # Open the Zarr file
    zfile = zarr.open(zarr_path, mode='r')

    # Get the class names
    class_names = zfile.attrs['labels']
    labels = {i: name for i, name in enumerate(class_names)}
    amp_params = zfile.attrs['amg']
    # convert to dict
    return labels, amp_params

def load_model(model, model_weights):
    # Freeze all parameters except classifier
    model.load_state_dict(torch.load(model_weights, weights_only=True))
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False 
    return model