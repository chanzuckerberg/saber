from saber.visualization import classifier as visualization
from contextlib import nullcontext
import torch.nn.functional as F
from lightning import Fabric
from saber.utils import io
from tqdm import tqdm
import os, zarr, yaml
import numpy as np
import torch

class ClassifierTrainerFabrics:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device=None,              # kept for API symmetry; Fabric manages devices
        beta: float = 1.0,
        include_background: bool = False,
        precision: str = "16-mixed",   # "16-mixed" or "32-true" also fine
        strategy: str = "ddp",
        devices: int | None = None,
        accelerator: str = "gpu",
    ):
        # --- Fabric init (spawns processes when needed) ---
        if devices is None:
            self.ngpus = torch.cuda.device_count() or 1
        else:
            self.ngpus = devices

        self.fabric = Fabric(
            precision=precision,
            strategy=strategy,
            devices=self.ngpus,
            accelerator=accelerator,
        )
        self.fabric.launch()

        # User-provided objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.beta = beta

        # Results containers (same format as your non-Fabric trainer)
        shared_keys = ['loss', 'recall', 'precision', 'f1_score', 'fbeta_score']
        self.results = {'train': {k: [] for k in shared_keys},
                        'val':   {k: [] for k in shared_keys}}

        self.num_classes = self.model.classifier[-1].out_features
        per_class_keys = ['recall', 'precision', 'f1_score', 'fbeta_score']
        self.per_class_results = {
            m: {f'class{i}': {k: [] for k in per_class_keys}
                for i in range(self.num_classes)}
            for m in ['train', 'val']
        }

        if include_background:
            self.indices = range(self.num_classes)
        else:
            self.indices = range(1, self.num_classes) if self.num_classes > 1 else [0]

        self.results_path = 'results'

    # ---- internal helpers -----------------------------------------------------

    def _fabric_setup(self, train_loader, val_loader):
        # Wrap model & optimizer
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Wrap dataloaders (injects DistributedSampler under DDP)
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

    def _mean_across_processes(self, value: float) -> float:
        t = torch.tensor([value], device=getattr(self.fabric, "device", "cpu"))
        self.fabric.all_reduce(t, reduce_op="mean")
        return float(t.item())

    # ---- core ops -------------------------------------------------------------

    def train_step(self, batch):
        # move once per step
        batch = self.fabric.to_device(batch)

        images = batch["image"]
        masks  = batch["mask"]
        labels = batch["label"]

        if labels.max() >= self.num_classes:
            labels = labels.clone()
            labels[labels >= self.num_classes] = 0

        with self.fabric.autocast():
            if getattr(self.model, "input_mode", None) == "concatenate":
                roi  = images * masks
                roni = images * (1 - masks)
                x = torch.cat([roi, roni], dim=1)
                logits = self.model(x)
            else:
                logits = self.model(images, masks)

            # If using CrossEntropyLoss, prefer index labels (faster than one-hot)
            # loss = self.loss_fn(logits, labels)
            labels_1h = F.one_hot(labels, num_classes=self.num_classes).float()
            loss = self.loss_fn(logits, labels_1h)

        self.optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(loss)
        self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        return float(loss.item()), preds.detach().cpu().tolist(), labels.detach().cpu().tolist()

    @torch.no_grad()
    def val_step(self, batch):
        # move once per step
        batch = self.fabric.to_device(batch)

        images = batch["image"]
        masks  = batch["mask"]
        labels = batch["label"]

        if labels.max() >= self.num_classes:
            labels = labels.clone()
            labels[labels >= self.num_classes] = 0

        # inference_mode is faster & more memory-efficient than no_grad
        with torch.inference_mode(), self.fabric.autocast():
            if getattr(self.model, "input_mode", None) == "concatenate":
                roi  = images * masks
                roni = images * (1 - masks)
                x = torch.cat([roi, roni], dim=1)
                logits = self.model(x)
            else:
                logits = self.model(images, masks)

            # If using CrossEntropyLoss: loss = self.loss_fn(logits, labels)
            labels_1h = F.one_hot(labels, num_classes=self.num_classes).float()
            loss = self.loss_fn(logits, labels_1h)

            preds = torch.argmax(logits, dim=1)

        return float(loss.item()), preds.cpu().tolist(), labels.cpu().tolist()

    # ---- public API (same as your non-Fabric trainer) -------------------------

    def train(self, train_loader, val_loader, num_epochs, best_metric='f1_score'):
        """
        Train the classifier using Lightning Fabric.
        """
        
        # Create results directory and Fabrics
        os.makedirs(self.results_path, exist_ok=True)
        self.fabric.seed_everything(42)
        self._fabric_setup(train_loader, val_loader)

        # Print training information
        if self.fabric.is_global_zero:
            print(f'Training with Lightning Fabric... (ngpus: {self.ngpus})')

        # Initialize the progress bar
        pbar = tqdm(
            total=num_epochs,
            desc="Training",
            disable=not self.fabric.is_global_zero,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        # Save model parameters
        self.save_parameters(num_epochs, self.train_loader.dataset.zarr_path, self.val_loader.dataset.zarr_path)
        
        # Initialize the best metric value for checkpointing
        best_metric_value = -1.0
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # ---- Train ----
            self.model.train()
            train_loss_sum = 0.0
            train_preds, train_labels = [], []
            
            for batch in self.train_loader:
                loss, preds, labels = self.train_step(batch)
                train_loss_sum += loss
                train_preds.extend(preds)
                train_labels.extend(labels)

            train_loss = self._mean_across_processes(
                train_loss_sum / max(1, len(self.train_loader))
            )
            self.results['train']['loss'].append(train_loss)
            
            # Gather all predictions across ranks
            train_preds_all = self._gather_all_predictions(train_preds)
            train_labels_all = self._gather_all_predictions(train_labels)
            self.store_metrics(train_preds_all, train_labels_all, 'train')

            # ---- Val ----
            self.model.eval()
            val_loss_sum = 0.0
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    loss, preds, labels = self.val_step(batch)
                    val_loss_sum += loss
                    val_preds.extend(preds)
                    val_labels.extend(labels)

            val_loss = self._mean_across_processes(
                val_loss_sum / max(1, len(self.val_loader))
            )
            self.results['val']['loss'].append(val_loss)
            
            # Gather all predictions across ranks
            val_preds_all = self._gather_all_predictions(val_preds)
            val_labels_all = self._gather_all_predictions(val_labels)
            self.store_metrics(val_preds_all, val_labels_all, 'val')

            # Scheduler step
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(self.results['val'][best_metric][-1])
            else:
                self.scheduler.step()

            # Checkpoint best
            current = self.results['val'][best_metric][-1]
            if current > best_metric_value:
                best_metric_value = current
                self._save_checkpoint(os.path.join(self.results_path, "best_model.pth"))

            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | {best_metric}: {current:.4f}",
                refresh=False
            )
            pbar.update(1)
        pbar.close()
        self.fabric.barrier()

    def _gather_all_predictions(self, local_list):
        """Gather predictions/labels from all ranks."""
        local_tensor = torch.tensor(local_list, dtype=torch.long)
        gathered = self.fabric.all_gather(local_tensor)
        
        if gathered.dim() > 1:
            gathered = gathered.flatten()
        
        return gathered.cpu().tolist()

    # ---- metrics / plotting (unchanged from your class) -----------------------

    def store_metrics(self, all_preds, all_labels, mode):
        precisions, recalls, f1s, fbetas = self.compute_metrics(all_preds, all_labels)
        macro_precision = sum(precisions[i] for i in self.indices) / len(self.indices)
        macro_recall = sum(recalls[i] for i in self.indices) / len(self.indices)
        macro_f1 = sum(f1s[i] for i in self.indices) / len(self.indices)
        macro_fbeta = sum(fbetas[i] for i in self.indices) / len(self.indices)
        self.results[mode]['precision'].append(macro_precision)
        self.results[mode]['recall'].append(macro_recall)
        self.results[mode]['f1_score'].append(macro_f1)
        self.results[mode]['fbeta_score'].append(macro_fbeta)
        for i in range(self.num_classes):
            k = f'class{i}'
            self.per_class_results[mode][k]['precision'].append(precisions[i])
            self.per_class_results[mode][k]['recall'].append(recalls[i])
            self.per_class_results[mode][k]['f1_score'].append(f1s[i])
            self.per_class_results[mode][k]['fbeta_score'].append(fbetas[i])

    def compute_metrics(self, preds, labels):
        tp = [0] * self.num_classes
        fp = [0] * self.num_classes
        fn = [0] * self.num_classes
        fbeta = [0] * self.num_classes
        for pred, label in zip(preds, labels):
            if pred == label:
                tp[label] += 1
            else:
                fp[pred] += 1
                fn[label] += 1
        per_p, per_r, per_f1, per_fb = [], [], [], []
        for i in range(self.num_classes):
            p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
            r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            fbeta[i] = (1 + self.beta**2) * (p * r) / ((self.beta**2 * p) + r) if ((self.beta**2 * p) + r) else 0.0
            per_p.append(p); per_r.append(r); per_f1.append(f1); per_fb.append(fbeta[i])
        return per_p, per_r, per_f1, per_fb

    def save_results(self, train_files, val_files, test_files=None):

        # Wait for all ranks to finish training
        self.fabric.barrier()

        # Only save on rank 0 to avoid file conflicts
        if self.fabric.is_global_zero:
            self.plot_metrics()
            
            z = zarr.open(os.path.join(self.results_path, 'classifier_metrics.zarr'), mode='w')
            for k, v in self.results.items():
                if isinstance(v, dict):
                    g = z.create_group(k)
                    for sk, sv in v.items():
                        g[sk] = np.array(sv, dtype=np.float32)
                else:
                    z[k] = np.array(v, dtype=np.float32)
            
            z['beta'] = np.array([self.beta], dtype=np.float32)
            
            # Convert to lists if needed
            if isinstance(train_files, str): 
                train_files = [train_files]
            if isinstance(val_files, str):   
                val_files = [val_files]
            if test_files is not None and isinstance(test_files, str): 
                test_files = [test_files]
            
            z['train_files'] = np.array(train_files, dtype=str)
            z['val_files'] = np.array(val_files, dtype=str)
            if test_files is not None:
                z['test_files'] = np.array(test_files, dtype=str)
            
            print("Results and dataset file lists saved to Zarr format.")

        # Wait for all ranks to finish training
        self.fabric.barrier()

    def plot_metrics(self):
        # Only plot on rank 0 to avoid file conflicts
        if self.fabric.is_global_zero:
            visualization.plot_all_metrics(
                self.results, 
                save_path=os.path.join(self.results_path, 'metrics.pdf')
            )
            visualization.plot_per_class_metrics(
                self.per_class_results, 
                save_path=os.path.join(self.results_path, 'per_class_metrics.pdf')
            )

    def _get_unwrapped_model(self):
        """Get the underlying model from Fabric's wrapper."""
        if hasattr(self.model, '_forward_module'):
            return self.model._forward_module
        elif hasattr(self.model, 'module'):
            return self.model.module
        else:
            return self.model

    def _save_checkpoint(self, path):
        """Save model checkpoint using Fabric, removing DDP prefixes."""
        unwrapped_model = self._get_unwrapped_model()
        state_dict = unwrapped_model.state_dict()
        
        # Strip 'module.' prefix from all keys (DDP artifact)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.fabric.save(path, {"model": clean_state_dict})

    def save_parameters(self, num_epochs, train_path, validate_path):
        """
        Save the model parameters to a YAML file.
        """

        # Get the metadata from the training dataset
        (labels, amg_params) = io.get_metadata(train_path)

        config = {
            'model': {
                'num_classes': self.num_classes,
                'weights': os.path.abspath(os.path.join(self.results_path, 'best_model.pth')),
            },
            'labels': labels,
            'data': {
                'train': train_path,
                'validate': validate_path
            },
            'amg_params': amg_params,            
            'optimizer': {
                'optimizer': self.optimizer.__class__.__name__,
                'scheduler': self.scheduler.__class__.__name__,
                'loss_fn': self.loss_fn.__class__.__name__, 
                'num_epochs': num_epochs
            },
        }
        
        os.makedirs(self.results_path, exist_ok=True)
        with open(os.path.join(self.results_path, 'model_config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)    