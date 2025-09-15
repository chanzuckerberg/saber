from saber.finetune.losses import MultiMaskIoULoss
from lightning import fabric
from tqdm import tqdm
import torch, os

class SAM2FinetuneTrainer:
    def __init__(self, predictor, train_loader, val_loader):

        # Store the predictor
        self.predictor = predictor

        # Two parameter groups for different LRs (optional)
        params = [
            {"params": [p for p in self.predictor.model.sam_mask_decoder.parameters() if p.requires_grad],
            "lr": 1e-4},
            {"params": [p for p in self.predictor.model.sam_prompt_encoder.parameters() if p.requires_grad],
            "lr": 5e-5},
        ]

        # Initialize the optimizer and dataloaders
        self.num_gpus = torch.cuda.device_count()
        self.fabric = fabric.Fabric(accelerator="cuda", strategy="ddp", devices=self.num_gpus)
        optimizer = torch.optim.AdamW(params, weight_decay=4e-5)
        self.predictor.model, self.optimizer = self.fabric.setup(self.predictor.model,optimizer)

        if val_loader is None:
            self.train_loader = self.fabric.setup_dataloaders(train_loader)
        else:
            self.train_loader, self.val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)

        # Initialize the loss function
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.supervise_all_iou = False
        self.iou_use_l1_loss = True

        # Initialize the use_boxes flag
        self.use_boxes = False

        # Initialize the save directory
        self.save_dir = 'results'
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def _stack_image_embeddings_from_predictor(self):
        """
        After predictor.set_image_batch(images), gather stacked image embeddings
        and high-res features for all B images.
        Returns:
            image_embeds: [B, C, H', W']
            hr_feats:     list[level] of [B, C, H', W']
        """
        # image_embed is a list[len=B] of [C, H', W']; stack to [B, C, H', W']
        image_embeds = torch.stack(list(self.predictor.model._features["image_embed"]), dim=0).to(self.fabric.device)

        # high_res_feats is a list[level], where each level is a list[len=B] of [C, H', W']
        hr = self.predictor.model._features["high_res_feats"]
        B = image_embeds.shape[0]
        hr_feats = [torch.stack([lvl[b] for b in range(B)], dim=0).to(self.fabric.device) for lvl in hr]
        return image_embeds, hr_feats

    def forward_step(self, batch):
        """
        Returns: prd_masks [N,K,H,W] logits, prd_scores [N,K], gt_masks [N,H,W], inst_img_ix [N]
        """
        images = batch["images"]  # list of HxWx3 uint8 or float; predictor handles them
        B = len(images)

        # 1) Encode images once
        self.predictor.set_image_batch(images)  # caches features on predictor
        image_embeds_B, hr_feats_B = self._stack_image_embeddings_from_predictor()

        # 2) Flatten instances across batch, move tensors to device
        inst_img_ix, gt_all, pts_all, lbl_all, box_all = [], [], [], [], []
        for b in range(B):
            for m, p, l, bx in zip(batch["masks"][b], batch["points"][b], batch["labels"][b], batch["boxes"][b]):
                inst_img_ix.append(b)
                gt_all.append(m.to(self.fabric.device))
                pts_all.append(p.to(self.fabric.device))
                lbl_all.append(l.to(self.fabric.device))
                box_all.append(bx.to(self.fabric.device))

        N = len(gt_all)
        if N == 0:
            return None, None, None, None
        inst_img_ix = torch.tensor(inst_img_ix, device=self.fabric.device, dtype=torch.long)

        # 3) Pad clicks to (N,P,2) and (N,P)
        P = max(p.shape[0] for p in pts_all)
        pts_pad = torch.zeros((N, P, 2), device=self.fabric.device, dtype=torch.float32)
        lbl_pad = torch.zeros((N, P), device=self.fabric.device, dtype=torch.float32)
        for i, (p, l) in enumerate(zip(pts_all, lbl_all)):
            pts_pad[i, :p.shape[0]] = p
            lbl_pad[i, :l.shape[0]] = l

        # Optional boxes
        boxes = torch.stack(box_all, dim=0) if (self.use_boxes and len(box_all) > 0) else None

        # 4) Prompt encoding
        mask_input, unnorm_coords, labels, _ = self.predictor._prep_prompts(
            input_point=pts_pad, input_label=lbl_pad, box=boxes, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=boxes if self.use_boxes else None,
            masks=None,
        )

        # 5) Gather per-instance image feats
        image_embeds = image_embeds_B[inst_img_ix]                   # [N,C,H',W']
        hr_feats = [lvl[inst_img_ix] for lvl in hr_feats_B]          # list of [N,C,H',W']

        # 6) Decode
        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=image_embeds,
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=hr_feats,
        )

        # 7) Upscale + stack GT
        prd_masks = self.predictor._transforms.postprocess_masks(
            low_res_masks, self.predictor._orig_hw[-1]
        )  # [N,K,H,W] logits
        gt_masks = torch.stack(gt_all, dim=0)                        # [N,H,W]

        return prd_masks, prd_scores, gt_masks, inst_img_ix

    @torch.no_grad()
    def validate_step(self, batch):
        """
        Validate the model on the given batch.
        """
        if 'embeddings' in batch:
            embeddings = batch['embeddings']
        else:
            self.predictor.set_image_batch(batch['images'])

        # Run AMG to get proposals
        # proposals = self.predictor

        metrics = {}
        return metrics

    def train(self, num_epochs):
        """
        Fine Tune SAM2 on the given data.
        """

        # Initialize the loss function
        self.loss_fn = MultiMaskIoULoss(
            weight_dict={"loss_mask": 1.0, "loss_dice": 1.0, "loss_iou": 0.05},
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            supervise_all_iou=self.supervise_all_iou,
            iou_use_l1_loss=self.iou_use_l1_loss
        )

        self.optimizer.zero_grad()
        for epoch in tqdm(range(num_epochs)):

            # Initialize the epoch loss
            epoch_loss_train = 0
            epoch_loss_val = 0

            # Train 
            self.predictor.model.train()
            for batch in self.train_loader:
                with self.fabric.autocast():
                    out = self.forward_step(batch)
                    if out[0] is None:
                        continue
                    prd_masks, prd_scores, gt_masks, _ = out
                    losses = self.loss_fn(prd_masks, prd_scores, gt_masks)
                self.fabric.backward(losses)
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss_train += float(losses["loss_total"].detach().cpu())

            # Validate
            self.predictor.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    with self.fabric.autocast():
                        out = self.validate_step(batch)
                        losses = self.loss_fn(out)

            # Print Only on Rank 0
            if self.fabric.is_global_zero:
                #Checkpoint 
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss_train/len(self.train_loader)}, Val Loss: {epoch_loss_val/len(self.val_loader)}")
                torch.save(self.predictor.model.state_dict(), f"{self.save_dir}/model.pth")