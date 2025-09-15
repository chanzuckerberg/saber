from lightning.fabric import Fabric
from tqdm import tqdm
import torch

class SAM2FinetuneTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Two parameter groups for different LRs (optional)
        params = [
            {"params": [p for p in model.sam_mask_decoder.parameters() if p.requires_grad],
            "lr": 1e-4},
            {"params": [p for p in model.sam_prompt_encoder.parameters() if p.requires_grad],
            "lr": 5e-5},
        ]

        self.optimizer = torch.optim.AdamW(params, weight_decay=4e-5)
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, train_loader, val_loader, num_epochs):
        """
        Fine Tune SAM2 on the given data.
        """

        best_metric_value = -1 
        for epoch in tqdm(range(num_epochs)):
            
            # Reset results for this epoch
            epoch_loss_train = 0
            epoch_loss_val = 0
