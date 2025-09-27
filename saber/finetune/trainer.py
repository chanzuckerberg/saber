from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from saber.finetune.helper import save_training_log
from saber.finetune.abiou import automask_metrics
from saber.finetune.losses import MultiMaskIoULoss
from saber.finetune.metrics import sam2_metrics
from lightning import fabric
import torch, os, optuna
from tqdm import tqdm

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
        optimizer = torch.optim.AdamW(params, weight_decay=1e-5)
        if self.num_gpus > 1:
            self.fabric = fabric.Fabric(accelerator="cuda", strategy="ddp", devices=self.num_gpus)
            self.fabric.launch()
            self.predictor.model, self.optimizer = self.fabric.setup(self.predictor.model,optimizer)
            self.predictor.model.mark_forward_method('forward_image')
            self.autocast = self.fabric.autocast
            self.use_fabric = True
        else:
            self.optimizer = optimizer
            self.use_fabric = False
            def _autocast():
                return torch.autocast(device_type="cuda", enabled=torch.cuda.is_available())
            self.autocast = _autocast
        self.device = next(self.predictor.model.parameters()).device

        # Setup dataloaders
        if self.use_fabric:
            self.train_loader, self.val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)
        else:
            self.train_loader, self.val_loader = train_loader, val_loader

        # Initialize the loss function
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.supervise_all_iou = True
        self.iou_use_l1_loss = True
        self.predict_multimask = True

        # Automask Generator Parameters
        self.amg_kwargs = dict(
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            stability_score_offset=0.0,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            box_nms_thresh=0.6,
            use_m2m=False,
            multimask_output=True,
        )
        self.nAMGtrials = 10

        # Initialize the use_boxes flag
        self.use_boxes = False

        # Initialize the save directory
        self.save_dir = 'results'
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def is_global_zero(self):
        # True on single-process runs; Fabric guards inside when present
        return (not self.use_fabric) or (self.use_fabric is not None and self.fabric.is_global_zero)

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
        pts_pad = torch.zeros((N, P, 2), device=self.device)
        lbl_pad = torch.full((N, P), -1.0, device=self.device)  # <- ignore
        for i,(p,l) in enumerate(zip(pts_all, lbl_all)):
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
        target_sizes = [self.predictor._orig_hw[int(b)] for b in inst_img_ix]  # list of (H, W), len = N
        # postprocess per instance (supports single size); do it in a loop then stack
        upsampled = []
        for i in range(low_res_masks.shape[0]):           # N
            H, W = target_sizes[i]
            up_i = self.predictor._transforms.postprocess_masks(
                low_res_masks[i:i+1], (H, W)
            )  # [1,K,H,W]
            upsampled.append(up_i)

        prd_masks = torch.cat(upsampled, dim=0)           # [N,K,H,W]
        gt_masks  = torch.stack(gt_all, dim=0).float()    # [N,H,W]

        return prd_masks, prd_scores, gt_masks, inst_img_ix

    @torch.no_grad()
    def validate_step(self, max_images=float('inf'), best_metric='ABIoU'):
        """
        Validate the model on the given batch.
        """

        # Set the model to evaluation mode
        self.predictor.model.eval()

        # Initialize the AMG
        amg = SAM2AutomaticMaskGenerator(
            model=self.predictor.model,
            **self.amg_kwargs
        )

        # --- local accumulators (tensor) ---
        loss_keys = ["loss_total", "loss_iou", "loss_dice", "loss_mask"]
        losses_sum = {k: torch.tensor(0.0, device=self.device) for k in loss_keys}
        n_inst     = torch.tensor(0.0, device=self.device)
        n_imgs     = torch.tensor(0.0, device=self.device)

        # Initialize the metrics sum
        metrics_sum = {k: torch.tensor(0.0, device=self.device) for k in self.metric_keys}

        num_images_seen = 0
        for batch in self.val_loader:

            # Compute Loss on decoder outputs
            out = self.forward_step(batch)
            if out[0] is None:  
                continue # no instances in this batch
            prd_masks, prd_scores, gt_masks = out[:3]
            local_n = torch.tensor(float(gt_masks.shape[0]), device=self.device)

            with self.autocast():
                batch_losses = self.loss_fn(prd_masks, prd_scores, gt_masks)

            if best_metric == 'ABIoU':
                m = automask_metrics(
                    self.predictor,                 # predictor or predictor.model (your function supports either)
                    batch["images"],                # list[H×W×3] or list[H×W]
                    batch["masks"],                 # list[list[H×W]]
                    top_k=20,
                    device=self.device,
                    autocast_ctx=self.autocast,
                    amg_kwargs=self.amg_kwargs,
                )
            else:
                m = sam2_metrics(batch, out, amg)

            # means → sums
            for k in loss_keys:
                # detach→cpu→float to avoid graph + dtype issues
                losses_sum[k] += float(batch_losses[k].detach().cpu()) * local_n
            n_inst += local_n

            # Weight by number of images so we can average correctly later
            img_count = float(m["num_images"])
            for k in self.metric_keys:
                metrics_sum[k] += torch.tensor(m[k] * img_count, device=self.device)
            n_imgs    += torch.tensor(img_count, device=self.device)

            num_images_seen += img_count
            if num_images_seen >= max_images:
                break

        # Reduce losses across ranks
        losses_sum = self._all_reduce_sum(losses_sum)
        n_inst     = self._all_reduce_sum(n_inst)
        n_imgs     = self._all_reduce_sum(n_imgs)
        metrics_sum = self._all_reduce_sum(metrics_sum)

        # Avoid divide-by-zero
        img_denom = max(n_imgs.item(), 1.0)
        inst_denom = max(n_inst.item(), 1.0)

        out = {
            "loss_total": (losses_sum["loss_total"] / inst_denom).item(),
            "loss_iou":   (losses_sum["loss_iou"]   / inst_denom).item(),
            "loss_dice":  (losses_sum["loss_dice"]  / inst_denom).item(),
            "loss_mask":  (losses_sum["loss_mask"]  / inst_denom).item(),
            "num_images": int(img_denom),
        }
        out.update({k: (metrics_sum[k] / img_denom).item() for k in self.metric_keys})

        return out

    def train(self, num_epochs, best_metric = 'ABIoU', resample_frequency = 100):
        """
        Fine Tune SAM2 on the given data.
        """

        # Initialize the loss function
        self.loss_fn = MultiMaskIoULoss(
            weight_dict={"loss_mask": 10.0, "loss_dice": 1.0, "loss_iou": 1.0},
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            supervise_all_iou=self.supervise_all_iou,
            iou_use_l1_loss=self.iou_use_l1_loss
        )

        # Initialize the metric keys
        if best_metric == 'ABIoU':
            self.metric_keys = ['ABIoU']
        else:
            self.metric_keys = ['prompt_miou', 'cal_mae', 'cal_brier', 'cal_ece', 'cal_tables', 
                           'AR', 'R@10', 'R@50', 'R@100']

        # Cosine scheduler w/Warmup ----
        # warmup_epochs = max(int(0.01 * num_epochs), 1)
        warmup_epochs = 5
        self.warmup_sched = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        self.cosine_sched = CosineAnnealingLR(self.optimizer, T_max=(num_epochs - warmup_epochs), eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [self.warmup_sched, self.cosine_sched], milestones=[warmup_epochs])

        # Progress bar only on rank 0
        if self.is_global_zero:
            pbar = tqdm(total=num_epochs, desc='Fine Tuning SAM2', unit='epoch', 
                        leave=True, dynamic_ncols=True)
        else:
            pbar = None

        self.optimizer.zero_grad()
        best_metric_value = float('-inf')
        # Main Loop
        for epoch in range(num_epochs):
            # Train 
            # at start of each epoch
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            if (epoch+1) % resample_frequency == 0:
                self.train_loader.dataset.resample_epoch()

            self.predictor.model.train()
            for batch in self.train_loader:
                out = self.forward_step(batch)
                if out[0] is None:
                    continue
                prd_masks, prd_scores, gt_masks = out[:3]
                with self.autocast():
                    batch_losses = self.loss_fn(prd_masks, prd_scores, gt_masks)
                if self.use_fabric:
                    self.fabric.backward(batch_losses['loss_total'])
                else:
                    batch_losses['loss_total'].backward()

                # number of instances this rank used to compute its per-batch means
                _local_n = gt_masks.shape[0]                    

                # (optional) gradient clip:
                if self.use_fabric:
                    # norm-based clipping (L2) on all params in the optimizer
                    self.fabric.clip_gradients(
                        self.predictor.model,
                        self.optimizer,
                        max_norm=1.0,         
                        norm_type=2.0
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.predictor.model.parameters(), 
                        1.0, norm_type=2.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                losses = self._reduce_losses(batch_losses, _local_n)

            # Learning Rate Scheduler
            self.scheduler.step()

            # Validate
            metrics = {}
            if (epoch+1) % 1e4 == 0:
                metrics['val'] = self.amg_param_tuner()
            else:  
                metrics['val'] = self.validate_step()
            metrics['train'] = losses

            # Print Only on Rank 0
            if self.is_global_zero:
                pbar.set_postfix({
                    "train_loss": f"{metrics['train']['loss_total']:.4f}",
                    "val_loss": f"{metrics['val']['loss_total']:.4f}",
                    f"val_{best_metric}": f"{metrics['val'][best_metric]:.4f}",
                })
                pbar.update(1)

                # Save Training Log
                metrics['epoch'] = epoch
                metrics['lr_mask'] = self.scheduler.get_last_lr()[0]
                metrics['lr_prompt'] = self.scheduler.get_last_lr()[1]
                save_training_log(metrics, self.save_dir, self.metric_keys)

                # Save Model if best metric is achieved
                ckpt = {"model": self.predictor.model.state_dict()}
                metric_value = metrics['val'].get(best_metric)
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    torch.save(ckpt, f"{self.save_dir}/best_model.pth")
                    print(f"Best {best_metric} saved!")
                else:
                    torch.save(ckpt, f"{self.save_dir}/bad_model.pth")

    def _reduce_losses(self, losses, num_elems: int = None):
        """
        Reduce the losses across ranks.
        """
        key_map = {
            "loss_iou": "loss_iou",
            "loss_dice": "loss_dice",
            "loss_mask": "loss_mask",
            "loss_total": "loss_total",
        }
        count = torch.tensor(float(num_elems if num_elems is not None else 1.0), device=self.device)
        out = {}
        if self.use_fabric:
            global_count = self.fabric.all_reduce(count, reduce_op="sum")
            for long_k, short_k in key_map.items():
                if long_k not in losses: 
                    continue
                num = torch.tensor(float(losses[long_k].detach().item()), device=self.device) * count
                global_num = self.fabric.all_reduce(num, reduce_op="sum")
                out[short_k] = (global_num / torch.clamp(global_count, min=1.0)).item()
        else:
            for long_k, short_k in key_map.items():
                if long_k in losses:
                    out[short_k] = float(losses[long_k].detach().item())
        return out

    def _all_reduce_sum(self, x):
        if not self.use_fabric:
            return x
        if isinstance(x, torch.Tensor):
            return self.fabric.all_reduce(x, reduce_op="sum")
        if isinstance(x, dict):
            return {k: self.fabric.all_reduce(v, reduce_op="sum") for k, v in x.items()}
        raise TypeError(f"_all_reduce_sum expects Tensor or dict[str,Tensor], got {type(x)}")

    # def _all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     """
    #     return self.fabric.all_reduce(x, reduce_op="sum") if self.use_fabric else x

############### Experimental - Automatic Mask Generator Tuning ###############

    def amg_param_tuner(self, n_trials=10):
        """
        Tune a few AMG thresholds with Bayesian optimization (TPE).
        Warm-start from the current self.amg_kwargs.
        """

        if self.use_fabric and not self.is_global_zero:
            # Non-zero ranks: wait for rank 0 to finish tuning and broadcast params.
            self.fabric.barrier()
            # Receive updated dict from rank 0
            self.amg_kwargs = self.fabric.broadcast(self.amg_kwargs, src=0)
            # Now run normal distributed validation so logs are comparable
            return self.validate_step()


        # Use a fixed sampler (seed for reproducibility)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))

        # ---- Warm start with current params ----
        # IMPORTANT: names must match suggest_* names in objective
        warm = {
            "pred_iou_thresh":        float(self.amg_kwargs["pred_iou_thresh"]),
            "stability_score_thresh": float(self.amg_kwargs["stability_score_thresh"]),
            "stability_score_offset": float(self.amg_kwargs["stability_score_offset"]),
        }
        study.enqueue_trial(warm)

        def objective(trial: optuna.Trial) -> float:
            """
            Objective for Optuna: maximize ABIoU on a held-out validation set,
            varying only a few AMG thresholds. Do NOT mutate self.amg_kwargs here.
            """
            # Suggest in sensible finetuned ranges
            pred_iou_thresh        = trial.suggest_float("pred_iou_thresh",        0.40, 0.75)
            stability_score_thresh = trial.suggest_float("stability_score_thresh", 0.55, 0.90)
            stability_score_offset = trial.suggest_float("stability_score_offset", 0.00, 0.30)

            # Build a LOCAL kwargs dict (copy), keep other knobs fixed
            amg_kwargs_trial = dict(self.amg_kwargs)
            amg_kwargs_trial.update({
                "pred_iou_thresh":        pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
                "stability_score_offset": stability_score_offset,
            })

            # Validate the model with the new AMG thresholds
            metrics = self.validate_step(
                amg_kwargs_trial, max_images=100, 
                reduce_all_ranks=not self.use_fabric
            )
            return metrics['ABIoU']

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Update trainer state with the best params
        best = study.best_params
        self.amg_kwargs.update({
            "pred_iou_thresh":        float(best["pred_iou_thresh"]),
            "stability_score_thresh": float(best["stability_score_thresh"]),
            "stability_score_offset": float(best["stability_score_offset"]),
        })

        # Let other ranks proceed and receive the dict
        if self.use_fabric:
            self.fabric.barrier()
            self.amg_kwargs = self.fabric.broadcast(self.amg_kwargs, src=0)


        if self.is_global_zero:
            print("AMG tuned →", {k: self.amg_kwargs[k] for k in ["pred_iou_thresh","stability_score_thresh","stability_score_offset"]})

        # Now run the normal distributed validate_step() so metrics are globally averaged
        return self.validate_step()