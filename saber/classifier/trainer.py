from saber.visualization import classifier as visualization
from monai.metrics import ConfusionMatrixMetric
import torch.nn.functional as F
import torch_ema as ema
import torch, zarr, os
from tqdm import tqdm
import numpy as np

class ClassifierTrainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler,
        loss_fn,
        device,
        beta = 1.0,
        include_background = False,
        use_ema: bool = False
    ):
        """
        Initialize the trainer for a classifier.
        """

        # 
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.beta = beta

        # Initialize results
        shared_keys = ['loss', 'recall', 'precision', 'f1_score', 'fbeta_score']   # Loss is shared between train and val
        self.results = {
            'train': { key: [] for key in shared_keys },
            'val': { key: [] for key in shared_keys }
        }

        # Per-class metrics using keys 'class1', 'class2', etc.
        per_class_keys = ['recall', 'precision', 'f1_score', 'fbeta_score']
        self.num_classes = self.model.classifier[-1].out_features
        self.per_class_results = {
            mode: { f'class{i}': { key: [] for key in per_class_keys } 
                    for i in range(self.num_classes) }
            for mode in ['train', 'val']
        }

        # Include background in metrics calculations
        if include_background:  self.indices = range(self.num_classes)
        else:                   self.indices = range(1, self.num_classes) if self.num_classes > 1 else [0]

        # Pre-Define Results path
        self.results_path = 'results'

        # Initialize EMAHandler for the model
        self.ema_experiment = use_ema
        if self.ema_experiment:
            self.ema_handler = ema.ExponentialMovingAverage(self.model.parameters(), decay=0.99)

    def process_batch(self, batch_data, mode=True):
        """
        Process a batch of data.
        """

        # Move data to device
        images = batch_data["image"].to(self.device)   # shape [B, 1, H, W]
        masks = batch_data["mask"].to(self.device)     # shape [B, 1, H, W]
        labels = batch_data["label"].to(self.device)   # shape [B]

        # Set values greater than self.num_classes to zero
        if labels.max() >= self.num_classes:
            labels[labels >= self.num_classes] = 0
    
        # Input for ConvNeXt and SwinTransformer Backbones
        if self.model.input_mode == 'concatenate':
            # (Option 1: Concatenate along the channel dimension)
            # x = torch.cat([images, masks], dim=1)     # shape [B, 2, H, W]

            # # (Option 2: Apply the mask to the image) - Create ROI and RONI
            roi = images * masks  # ROI: Region of Interest
            roni = images * (1 - masks)  # RONI: Region of Non-Interest

            # # Concatenate ROI and RONI along the channel dimension
            x = torch.cat([roi, roni], dim=1)  # shape [B, 2, H, W]

            # Forward pass
            logits = self.model(x)
        # Input for cryoDinov2 and SAM2 Backbones
        else:
            # Forward pass
            logits = self.model(images, masks)

        # Convert labels to one-hot encoding
        labels = F.one_hot(labels, num_classes=self.num_classes).float()  # shape [B, num_classes]        
        
        # Compute loss
        loss = self.loss_fn(logits, labels)

        # Backpropagation
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update EMA weights
            if self.ema_experiment:
                self.ema_handler.update()

        # Predictions
        preds = torch.argmax(logits, dim=1)

        # Convert labels to one-hot encoding
        labels = torch.argmax(labels, dim=-1)

        return loss.item(), preds, labels

    def store_metrics(self, all_preds, all_labels, mode):
        """
        Store metrics for a given mode.
        """
        precisions, recalls, f1s, fbetas = self.compute_metrics(all_preds, all_labels)

        # Compute macro-averaged precision, recall, and F1 score.
        macro_precision = sum(precisions[i] for i in self.indices) / len(self.indices)
        macro_recall = sum(recalls[i] for i in self.indices) / len(self.indices)
        macro_f1 = sum(f1s[i] for i in self.indices) / len(self.indices)
        macro_fbeta = sum(fbetas[i] for i in self.indices) / len(self.indices)

        # Store macro-averaged metrics.
        self.results[mode]['precision'].append(macro_precision)
        self.results[mode]['recall'].append(macro_recall)
        self.results[mode]['f1_score'].append(macro_f1)
        self.results[mode]['fbeta_score'].append(macro_fbeta)

        # Store per-class metrics. We generate keys 'class1', 'class2', etc.
        for i in range(self.num_classes):
            cls_key = f'class{i}'
            self.per_class_results[mode][cls_key]['precision'].append(precisions[i])
            self.per_class_results[mode][cls_key]['recall'].append(recalls[i])
            self.per_class_results[mode][cls_key]['f1_score'].append(f1s[i])
            self.per_class_results[mode][cls_key]['fbeta_score'].append(fbetas[i])

    def train(self, train_loader, val_loader, num_epochs, best_metric = 'f1_score'):
        """
        Train the classifier.
        """

        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)

        best_metric_value = -1 
        for epoch in tqdm(range(num_epochs)):
            
            # Reset results for this epoch
            epoch_loss_train = 0
            epoch_loss_val = 0

            # Training
            all_preds = []; all_labels = []
            self.model.train()
            for batch_data in train_loader:
                loss, preds, labels = self.process_batch(batch_data, mode='train')
                epoch_loss_train += loss
                # Move predictions and labels to CPU and store as Python lists
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
            self.results['train']['loss'].append((epoch_loss_train/len(train_loader)))

            # Compute training metrics
            self.store_metrics(all_preds, all_labels, 'train')

            # Validation
            all_preds = []; all_labels = []
            self.model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    if self.ema_experiment:
                        with self.ema_handler.average_parameters():
                            loss, preds, labels = self.process_batch(batch_data, mode='val')
                    else:
                        loss, preds, labels = self.process_batch(batch_data, mode='val')
                    epoch_loss_val += loss
                    # Move predictions and labels to CPU and store as Python lists
                    all_preds.extend(preds.detach().cpu().tolist())
                    all_labels.extend(labels.detach().cpu().tolist())
            self.results['val']['loss'].append((epoch_loss_val/len(val_loader)))

            # Compute training metrics
            self.store_metrics(all_preds, all_labels, 'val')

            # Adjust learning rate
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(self.results['val'][best_metric][-1])
            else:
                self.scheduler.step()

            # Save the best model
            if self.results['val'][best_metric][-1] > best_metric_value:
                best_metric_value = self.results['val'][best_metric][-1]

                if self.ema_experiment:
                    with self.ema_handler.average_parameters():
                        torch.save(self.model.state_dict(), os.path.join(self.results_path, "best_model.pth"))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.results_path, "best_model.pth"))
                print("Model saved!")

            tqdm.write(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss_train/len(train_loader)}")

    # Save Training Results to Zarr File
    def save_results(self, train_files, val_files, test_files = None):

        # Save a PDF with Metrics
        self.plot_metrics()

        # Save results as Zarr
        zarr_results = zarr.open(os.path.join(self.results_path,'classifier_metrics.zarr'), mode='w')
        for key, value in self.results.items():
            if isinstance(value, dict):
                group = zarr_results.create_group(key)
                for sub_key, sub_value in value.items():
                    group[sub_key] = np.array(sub_value).astype(np.float32)
            else:
                zarr_results[key] = np.array(value).astype(np.float32)

        # Store the beta value used for training
        zarr_results['beta'] = np.array([self.beta]).astype(np.float32)                

        # Ensure train_files and val_files are lists
        if isinstance(train_files, str):
            train_files = [train_files]
        if isinstance(val_files, str):
            val_files = [val_files]
        if test_files is not None and isinstance(test_files, str):
            test_files = [test_files]

         # Save dataset file lists
        zarr_results['train_files'] = np.array(train_files, dtype=str)
        zarr_results['val_files'] = np.array(val_files, dtype=str)

        if test_files is not None:
            zarr_results['test_files'] = np.array(test_files, dtype=str)

        print("Results and dataset file lists saved to Zarr format.") 

    def compute_metrics(self, preds, labels):
        """
        Compute macro-averaged precision, recall, and F1 score.
        
        Returns:
            macro_precision (float), macro_recall (float), macro_f1 (float)
        """
        # Initialize counts for each class
        tp = [0] * self.num_classes  # True positives
        fp = [0] * self.num_classes  # False positives
        fn = [0] * self.num_classes  # False negatives
        fbeta = [0] * self.num_classes  # F-beta score

        # Count TP, FP, and FN per class
        for pred, label in zip(preds, labels):
            if pred == label:
                tp[label] += 1
            elif pred != label and pred > 0:
                fp[pred] += 1
            elif pred != label and pred == 0:
                fn[label] += 1

        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        per_class_fbeta = []
        
        for i in range(self.num_classes):
            # Avoid division by zero
            p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            fbeta[i] = (1 + self.beta**2) * (p * r) / ((self.beta**2 * p) + r) if ((self.beta**2 * p) + r) > 0 else 0.0
            
            per_class_precision.append(p)
            per_class_recall.append(r)
            per_class_f1.append(f1) 
            per_class_fbeta.append(fbeta[i])

        return per_class_precision, per_class_recall, per_class_f1, per_class_fbeta

    def plot_metrics(self):
        """
        Plot the metrics.
        """
        visualization.plot_all_metrics(self.results, 
                                       save_path=os.path.join(self.results_path,'metrics.pdf'))
        visualization.plot_per_class_metrics(self.per_class_results, 
                                             save_path=os.path.join(self.results_path,'per_class_metrics.pdf'))
