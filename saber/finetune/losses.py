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
    def __init__(self, weight_dict, focal_alpha=0.25, focal_gamma=2.0,
                 supervise_all_iou=True, iou_use_l1_loss=True, all_iou_weight=0.1):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.supervise_all_iou = supervise_all_iou
        self.all_iou_weight = all_iou_weight
        self.iou_use_l1_loss = iou_use_l1_loss

    def forward(self, prd_masks, prd_scores, gt_masks):
        # prd_masks: [N,K,H,W] (logits), prd_scores: [N,K] (IoU logits), gt_masks: [N,H,W] (0/1)
        device = prd_masks.device
        N, K, H, W = prd_masks.shape
        gt_masks = gt_masks.to(prd_masks.dtype)

        # --- compute per-proposal seg losses (no reduction) ---
        gt_rep = gt_masks.repeat_interleave(K, dim=0)                  # [N*K,H,W]
        focal_per_k = focal_loss_from_logits(
            prd_masks.view(N*K, H, W), gt_rep,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        ).view(N, K)                                                   # [N,K]
        dice_per_k = dice_loss_from_logits(
            prd_masks.view(N*K, H, W), gt_rep
        ).view(N, K)                                                   # [N,K]
        seg_loss_per_k = focal_per_k + dice_per_k

        # --- choose the slot by lowest seg loss (SAM2-style) ---
        best_idx = seg_loss_per_k.argmin(dim=1)                         # [N]
        row = torch.arange(N, device=device)
        logits_star = prd_masks[row, best_idx]                          # [N,H,W]

        # --- actual IoU per proposal (stop grad) ---
        with torch.no_grad():
            pred_bin = (prd_masks > 0).to(gt_masks.dtype)               # [N,K,H,W]
            gt_k = gt_masks[:, None].expand_as(pred_bin)                # [N,K,H,W]
            inter = (pred_bin * gt_k).sum(dim=(2,3))
            union = (pred_bin + gt_k - pred_bin*gt_k).sum(dim=(2,3)).clamp_min(1e-6)
            true_iou_k = inter / union                                  # [N,K]
        true_iou_star = true_iou_k[row, best_idx]

        # --- seg losses on the chosen slot only ---
        l_focal = focal_loss_from_logits(logits_star, gt_masks,
                                         alpha=self.focal_alpha, gamma=self.focal_gamma).mean()
        l_dice  = dice_loss_from_logits(logits_star, gt_masks).mean()

        # --- IoU head regression ---
        pred_iou = prd_scores.sigmoid()
        if self.iou_use_l1_loss:
            l_iou = F.l1_loss(pred_iou[row, best_idx], true_iou_star)
        else:
            l_iou = F.mse_loss(pred_iou[row, best_idx], true_iou_star)

        if self.supervise_all_iou:
            if self.iou_use_l1_loss:
                l_iou_all = F.l1_loss(pred_iou, true_iou_k)
            else:
                l_iou_all = F.mse_loss(pred_iou, true_iou_k)
            l_iou = l_iou + self.all_iou_weight * l_iou_all

        # --- weighted sum ---
        loss_mask = l_focal
        loss_dice = l_dice
        loss_iou  = l_iou
        total = (self.weight_dict["loss_mask"] * loss_mask
               + self.weight_dict["loss_dice"] * loss_dice
               + self.weight_dict["loss_iou"]  * loss_iou)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice, "loss_iou": loss_iou, "loss_total": total}