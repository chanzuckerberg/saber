# Training a Domain Expert Classifier

This guide explains how to train a domain expert classifier for use with SABER's segmentation pipeline. Custom classifiers can significantly improve segmentation quality by filtering out irrelevant segments and focusing on structures of interest.

## 🎯 What You'll Learn

* Prepare training data from segmentation results
* Configure and train a classifier

## Step 1: Import Modules and Dataloader
Before training, you need properly structured data in Zarr format. SABER provides tools to create training datasets from segmentation results:

```python

from saber.classifier.datasets import singleZarrDataset, augment
from saber.classifier.trainer import ClassifierTrainer
from saber.classifier.models import common
from saber.utils import io

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import FocalLoss
from monai.transforms import Compose
import torch, yaml, os

# Training settings
train_path = "path/to/training_data.zarr" 
val_path = 'path/to/validation_data.zarr'  
num_epochs = 100   # Number of training epoch
num_classes = 2    # Binary classification (background + target)

# Set device
device = io.get_available_devices()

# Create dataloader with appropriate settings
transforms = Compose([augment.get_preprocessing_transforms(True), augment.get_training_transforms()])
dataset = singleZarrDataset.ZarrSegmentationDataset(train_path, mode=mode, transform=transforms)
train_loader = DataLoader( dataset, batch_size=batch_size,  shuffle=True, drop_last=True)

transforms = Compose([augment.get_validation_transforms()])
dataset = singleZarrDataset.ZarrSegmentationDataset(val_path, mode=mode, transform=transforms)
train_loader = DataLoader( dataset, batch_size=batch_size,  shuffle=False, drop_last=True)
```

## Step 2: Initialize Model, Optimizer and Loss Function

```python
# Initialize model
model_size = 'base'
model = common.get_classifier_model('SAM2', num_classes, model_size)
model = model.to(device)

# Configure optimizer 
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Loss function - FocalLoss works well for imbalanced datasets
loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
```

## Step 3: Train the Model

SABER provides a `ClassifierTrainer` class that handles the training loop:
```python
# Initialize trainer
trainer = ClassifierTrainer(model, optimizer, scheduler, loss_fn, device)

# Train the model
trainer.train(train_loader, val_loader, num_epochs)

# Save results and performance metrics
trainer.save_results(train_path, validate_path)
```
In the output directory (default is `results/`), saber will provide plots of the average and per class training metrics in a saved pdf file. 

## Step 4: Save the Model Configuration

Save the model configuration for easy reuse with SABER's segmenters:
```python
# Save model configuration
model_config = {
    'model': { 
        'backbone': backbone, 'model_size': model_size, 'num_classes': num_classes,
        'weights': os.path.abspath(os.path.join(results_path, 'best_model.pth'))
    },
    'data': { 'train': train_path, 'validate': validate_path }
}

# Save configuration to YAML file
config_path = os.path.join(results_path, 'model_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False, sort_keys=False, indent=2)
```

## 📚 Next Steps

Now that we have a trained model, we can use this classifier to generate object specific segmentations in our 2D and 3D datasets!

* [2D Micrograph Segmentation](quickstart2d.md) - Use your trained classifier with 2D data
* [3D Tomogram Segmentation](quickstart3d.md) - Apply your classifier to 3D volumes
* [Parallel Inference](parallel-inference.md) - Scale up processing with GPU parallelization