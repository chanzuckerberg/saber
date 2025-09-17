from saber.finetune.metrics import automask_metrics
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
        optimizer = torch.optim.AdamW(params, weight_decay=4e-5)
        if self.num_gpus > 1:
            self.fabric = fabric.Fabric(accelerator="cuda", strategy="ddp", devices=self.num_gpus)
            self.fabric.launch()
            self.predictor.model, self.optimizer = self.fabric.setup(self.predictor.model,optimizer)
            self.autocast = self.fabric.autocast
            self.use_fabric = True
        else:
            self.optimizer = optimizer
            self.use_fabric = False
            self.autocast = torch.cuda.amp.autocast
        self.device = next(self.predictor.model.parameters()).device

        if val_loader is None and self.use_fabric:
                self.train_loader = self.fabric.setup_dataloaders(train_loader)
        elif self.use_fabric and val_loader is not None:
            self.train_loader, self.val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)
        else:
            self.train_loader, self.val_loader = train_loader, val_loader

        # Initialize the loss function
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.supervise_all_iou = False
        self.iou_use_l1_loss = True
        self.predict_multimask = True

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
        image_embeds = torch.stack(list(self.predictor._features["image_embed"]), dim=0).to(self.device)

        # high_res_feats is a list[level], where each level is a list[len=B] of [C, H', W']
        hr = self.predictor._features["high_res_feats"]
        B = image_embeds.shape[0]
        hr_feats = [torch.stack([lvl[b] for b in range(B)], dim=0).to(self.device) for lvl in hr]
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
                gt_all.append(m.to(self.device))
                pts_all.append(p.to(self.device))
                lbl_all.append(l.to(self.device))
                box_all.append(bx.to(self.device))

        N = len(gt_all)
        if N == 0:
            return None, None, None, None
        inst_img_ix = torch.tensor(inst_img_ix, device=self.device, dtype=torch.long)

        # 3) Pad clicks to (N,P,2) and (N,P)
        P = max(p.shape[0] for p in pts_all)
        pts_pad = torch.zeros((N, P, 2), device=self.device, dtype=torch.float32)
        lbl_pad = torch.zeros((N, P), device=self.device, dtype=torch.float32)
        for i, (p, l) in enumerate(zip(pts_all, lbl_all)):
            pts_pad[i, :p.shape[0]] = p
            lbl_pad[i, :l.shape[0]] = l

        # Optional boxes
        boxes = torch.stack(box_all, dim=0) if (self.use_boxes and len(box_all) > 0) else None

        # 4) Prompt encoding
        mask_input, unnorm_coords, labels, _ = self.predictor._prep_prompts(
            pts_pad, lbl_pad, 
            box=boxes, mask_logits=None, 
            normalize_coords=True
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
            multimask_output=self.predict_multimask,
            repeat_image=False,
            high_res_features=hr_feats,
        )

        # 7) Upscale + stack GT
        prd_masks = self.predictor._transforms.postprocess_masks(
            low_res_masks, self.predictor._orig_hw[-1]
        )  # [N,K,H,W] logits
        gt_masks = torch.stack(gt_all, dim=0).float()                       # [N,H,W]

        return prd_masks, prd_scores, gt_masks, inst_img_ix

    @torch.no_grad()
    def validate_step(self):
        """
        Validate the model on the given batch.
        """

        self.predictor.model.eval()

        # Local accumulators (weighted by number of images in each call)
        abiou_sum = torch.tensor(0.0, device=self.device)
        map_sum   = torch.tensor(0.0, device=self.device)
        n_imgs    = torch.tensor(0.0, device=self.device)

        # Each rank iterates only its shard (Fabric sets DistributedSampler for you)
        for batch in self.val_loader:
            # Compute metrics on THIS batch only (keeps memory small & parallel)
            m = automask_metrics(
                self.predictor,                 # predictor or predictor.model (your function supports either)
                batch["images"],                # list[H×W×3] or list[H×W]
                batch["masks"],                 # list[list[H×W]]
                amg_kwargs={"points_per_side": 16, "pred_iou_thresh": 0.7, "crop_n_layers": 1},
                top_k=100,
                compute_map=True,
                device=self.device
            )

            # Weight by number of images so we can average correctly later
            num = float(m["num_images"])
            abiou_sum += torch.tensor(m["ABIoU"] * num, device=self.device)
            if m["mAP"] is not None:
                map_sum += torch.tensor(m["mAP"] * num, device=self.device)
            n_imgs    += torch.tensor(num, device=self.device)

        # Global reduction (sum across all ranks)
        if self.use_fabric:
            abiou_sum = self.fabric.all_reduce(abiou_sum, reduce_op="sum")
            map_sum   = self.fabric.all_reduce(map_sum,   reduce_op="sum")
            n_imgs    = self.fabric.all_reduce(n_imgs,    reduce_op="sum")
        else:
            abiou_sum = abiou_sum.sum()
            map_sum   = map_sum.sum()
            n_imgs    = n_imgs.sum()

        # Avoid divide-by-zero
        denom = max(n_imgs.item(), 1.0)
        return {
            "ABIoU": (abiou_sum / denom).item(),
            "mAP":   (map_sum   / denom).item(),
            "num_images": int(denom),
        }

    def train(self, num_epochs, best_metric = 'mAP'):
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

        best_metric_value = float('-inf')
        self.optimizer.zero_grad()
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            # Train 
            epoch_loss_train = 0
            self.predictor.model.train()
            self.train_loader.dataset.resample_epoch()
            for batch in self.train_loader:
                out = self.forward_step(batch)
                if out[0] is None:
                    continue
                prd_masks, prd_scores, gt_masks, _ = out
                with self.autocast():
                    losses = self.loss_fn(prd_masks, prd_scores, gt_masks)
                if self.use_fabric:
                    self.fabric.backward(losses['loss_total'])
                else:
                    losses['loss_total'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss_train += float(losses["loss_total"].detach().cpu())

            import pdb; pdb.set_trace()

            # Validate
            metrics = self.validate_step()

            import pdb; pdb.set_trace()

            # Print Only on Rank 0
            if self.fabric.is_global_zero:
                print(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"Loss={epoch_loss_train/len(self.train_loader):.5f} "
                    f"mAP={metrics['mAP']:.4f} - ABIoU={metrics['ABIoU']:.4f} "
                )

                if metrics[best_metric] > best_metric_value:
                    best_metric_value = metrics[best_metric]
                    torch.save(self.predictor.model.state_dict(), f"{self.save_dir}/best_model.pth")
                    print(f"Best {best_metric} saved!")