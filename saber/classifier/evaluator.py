from saber.classifier.datasets import singleZarrDataset, augment
from saber.classifier.trainer import ClassifierTrainer
from saber.classifier.models import common
from torch.utils.data import DataLoader
from monai.transforms import Compose
import torch, click, yaml, os, csv
import torch.nn.functional as F
from saber.utils import io
from tqdm import tqdm
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

class ClassifierEvaluator(ClassifierTrainer):
    def __init__(self, dataset, model, device, beta=1.0, include_background=False):
        
        dummy_loss_fn = lambda x, y: torch.tensor(0.0)
        
        # Call parent constructor with dummy optimizer/scheduler
        super().__init__(
            model=model,
            optimizer=None,
            scheduler=None,
            loss_fn=dummy_loss_fn,
            device=device,
            beta=beta,
            include_background=include_background
        )
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Create Dataloader
        transforms = Compose([augment.get_validation_transforms()])
        dataset = singleZarrDataset.ZarrSegmentationDataset(dataset, mode='val', transform=transforms)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    def evaluate(self, output_file = None):
        """
        Evaluate the model and optionally save results to a file.
        
        Args:
            output_file (str, optional): Path to save results. If None, only prints results.
            format (str): Output format - 'json', 'yaml', or 'csv'
            
        Returns:
            dict: Dictionary with evaluation metrics
        """        
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc="Evaluating"):
                loss, preds, labels = self.process_batch(batch_data, mode='val')
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

        # Use the inherited compute_metrics method
        precisions, recalls, f1s, fbetas = self.compute_metrics(all_preds, all_labels)
        
        # Calculate Macro Metrics
        macro_fbeta = sum(fbetas[i] for i in self.indices) / len(self.indices)
        macro_recall = sum(recalls[i] for i in self.indices) / len(self.indices)
        macro_precision = sum(precisions[i] for i in self.indices) / len(self.indices)
        macro_f1 = sum(f1s[i] for i in self.indices) / len(self.indices)

        # Create results dictionary
        results = {
            'beta': float(self.beta),
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1),
            'fbeta': float(macro_fbeta)
        }
        
        # Save results if output_file is provided
        if output_file:
            if os.path.dirname(output_file) != '' :
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])  # Header
                for metric, value in results.items():
                    writer.writerow([metric, value])
                
            print(f"Results saved to {output_file}")
        else:
            # Print results
            print(f"Evaluation Results:")
            print(f"F-beta Score (β={self.beta}): {macro_fbeta:.4f}")
            print(f"Precision: {macro_precision:.4f}")
            print(f"Recall: {macro_recall:.4f}")

        return results

@click.command(context_settings={"show_default": True})
@click.option('--test', type=str, required=True, help='Path to the Test Zarr File')
@click.option('--model-config', type=str, required=True, help='Path to the Model Config')
@click.option('--model-weights', type=str, required=True, help='Path to the Model Weights')
@click.option('--beta', type=float, default=1.0, help='Beta for the F-Beta Score')
@click.option('--output', type=str, default=None, help='Path to save results')
def evaluate(test, model_config, model_weights, beta, output):
    """
    Evaluate a classifier model on a test set.
    """

    # Check if test file exists
    if not os.path.exists(test):
        raise FileNotFoundError(f"Test Zarr file {test} does not exist.")

    # Check if output file has a .csv extension
    if output is not None and output[-3:] != 'csv':
        raise ValueError(f"Output file must have a .csv extension. Received {output}.")

    # Load Model Config
    if os.path.exists(model_weights) and os.path.exists(model_config):
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Model config file {model_config} or model weights file {model_weights} does not exist.")            
            
    # Get Model Checkpoint
    model = common.get_classifier_model(
        config['model']['backbone'], 
        config['model']['num_classes'], 
        config['model']['model_size']  )    
    model.load_state_dict(torch.load(model_weights, weights_only=True))

    # Create the Evaluator
    device = io.get_available_devices()
    model = model.to(device)
    evaluator = ClassifierEvaluator(test, model, device, beta)

    # Evaluate the Model
    evaluator.evaluate(output)


