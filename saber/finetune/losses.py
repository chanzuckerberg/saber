import torch.nn.functional as F
import torch.nn as nn
import torch

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return 1 - (2 * inter + eps) / (denom + eps)

def focal_loss_from_logits(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (w * ((1 - pt) ** gamma) * ce).mean(dim=(1, 2))

class MultiMaskIoULoss(nn.Module):
    """
    General loss for multi-mask predictions with IoU calibration.
    Designed for SAM/SAM2 fine-tuning with AMG.
    """

    def __init__(self,
                 weight_dict: dict,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 supervise_all_iou=False,
                 iou_use_l1_loss=True):
        super().__init__()
        self.weight_dict = weight_dict
        assert "loss_mask" in weight_dict
        assert "loss_dice" in weight_dict
        assert "loss_iou" in weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss

    def forward(self, prd_masks, prd_scores, gt_masks):
        """
        Args
        ----
        prd_masks: [N, K, H, W] logits from decoder
        prd_scores: [N, K] predicted IoU logits (will be sigmoided here)
        gt_masks: [N, H, W] float {0,1}
        """

        device = prd_masks.device
        N, K, H, W = prd_masks.shape
        gt_masks = gt_masks.to(prd_masks.dtype)                 # [N,H,W]

        # ---- Compute hard predictions and true IoU per proposal (no grad) ----------
        with torch.no_grad():
            pred_bin  = (prd_masks > 0.0).to(gt_masks.dtype)      # [N,K,H,W]
            gt_k      = gt_masks[:, None].expand_as(pred_bin)   # [N,K,H,W]
            inter     = (pred_bin * gt_k).sum(dim=(2, 3))       # [N,K]
            union     = (pred_bin + gt_k - pred_bin * gt_k).sum(dim=(2, 3)).clamp_min(1e-6)
            true_iou_k = inter / union                          # [N,K] in [0,1]

        # ---- Per-proposal segmentation loss (focal + dice), select argmin ----------
        gt_rep = gt_masks.repeat_interleave(K, dim=0)           # [N*K,H,W]
        focal_per_k = focal_loss_from_logits(
            prd_masks.view(N*K, H, W), gt_rep,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        ).view(N, K)                                            # [N,K]
        dice_per_k = dice_loss_from_logits(
            prd_masks.view(N*K, H, W), gt_rep
        ).view(N, K)                                            # [N,K]

        seg_loss_per_k = focal_per_k + dice_per_k               # [N,K]
        # best_idx = seg_loss_per_k.argmin(dim=1)                 # [N] choose lowest seg loss
        best_idx = true_iou_k.argmax(dim=1)                 # [N] choose highest IoU

        row = torch.arange(N, device=device)
        logits_star = prd_masks[row, best_idx]                  # [N,H,W]
        true_iou_star = true_iou_k[row, best_idx].detach()      # [N]

        # ---- Segmentation losses on the chosen proposal ----------------------------
        l_focal = focal_loss_from_logits(
            logits_star, gt_masks,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        ).mean()                                                # scalar
        l_dice  = dice_loss_from_logits(logits_star, gt_masks).mean()

        # ---- IoU head regression (sigmoid + L1 by default) -------------------------
        pred_iou = prd_scores.sigmoid()                         # [N,K] in [0,1]

        if self.iou_use_l1_loss:
            l_iou = F.l1_loss(pred_iou[row, best_idx], true_iou_star)
        else:
            l_iou = F.mse_loss(pred_iou[row, best_idx], true_iou_star)

        # Optional: supervise IoU for *all* proposals with small weight
        if self.supervise_all_iou:
            if self.iou_use_l1_loss:
                l_iou_all = F.l1_loss(pred_iou, true_iou_k.detach())
            else:
                l_iou_all = F.mse_loss(pred_iou, true_iou_k.detach())
            l_iou = l_iou + 0.1 * l_iou_all   # tune 0.05–0.2 if needed

        # ---- Weighted sum ----------------------------------------------------------
        loss_mask = l_focal
        loss_dice = l_dice
        loss_iou  = l_iou

        total_loss = (self.weight_dict["loss_mask"] * loss_mask +
                    self.weight_dict["loss_dice"] * loss_dice +
                    self.weight_dict["loss_iou"]  * loss_iou)

        return {
            "loss_mask":  loss_mask,
            "loss_dice":  loss_dice,
            "loss_iou":   loss_iou,
            "loss_total": total_loss,
        }