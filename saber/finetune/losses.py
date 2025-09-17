import torch
import torch.nn as nn
import torch.nn.functional as F

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
        prd_scores: [N, K] predicted IoU scores
        gt_masks: [N, H, W] float {0,1}
        """
        N, K, H, W = prd_masks.shape

        # compute per-proposal losses
        loss_mask_k, loss_dice_k = [], []
        for k in range(K):
            l_focal = focal_loss_from_logits(
                prd_masks[:, k], gt_masks,
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )  # scalar over batch
            l_dice = dice_loss_from_logits(prd_masks[:, k], gt_masks)  # [N]
            loss_mask_k.append(l_focal.expand_as(l_dice))
            loss_dice_k.append(l_dice)
        loss_mask_k = torch.stack(loss_mask_k, dim=1)   # [N,K]
        loss_dice_k = torch.stack(loss_dice_k, dim=1)   # [N,K]

        # combine to pick best proposal per instance
        combo = (self.weight_dict["loss_mask"] * loss_mask_k +
                 self.weight_dict["loss_dice"] * loss_dice_k)
        best_idx = combo.argmin(dim=1)  # [N]
        row = torch.arange(N, device=prd_masks.device)

        # select best proposal losses
        loss_mask = loss_mask_k[row, best_idx].mean()
        loss_dice = loss_dice_k[row, best_idx].mean()

        # IoU calibration loss
        with torch.no_grad():
            probs = torch.sigmoid(prd_masks[row, best_idx])   # [N,H,W]
            pred_bin = (probs > 0.5).float()
            inter = (gt_masks * pred_bin).sum(dim=(1, 2))
            union = gt_masks.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) - inter + 1e-6
            true_iou = inter / union                          # [N]

        if self.supervise_all_iou:
            # supervise all proposals
            iou_targets = []
            for k in range(K):
                probs = torch.sigmoid(prd_masks[:, k])
                pred_bin = (probs > 0.5).float()
                inter = (gt_masks * pred_bin).sum(dim=(1, 2))
                union = gt_masks.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) - inter + 1e-6
                iou_targets.append(inter / union)
            iou_targets = torch.stack(iou_targets, dim=1)  # [N,K]
            if self.iou_use_l1_loss:
                loss_iou = F.l1_loss(prd_scores, iou_targets)
            else:
                loss_iou = F.mse_loss(prd_scores, iou_targets)
        else:
            score_best = prd_scores[row, best_idx]          # [N]
            if self.iou_use_l1_loss:
                loss_iou = F.l1_loss(score_best, true_iou)
            else:
                loss_iou = F.mse_loss(score_best, true_iou)

        # weighted sum
        total_loss = (self.weight_dict["loss_mask"] * loss_mask +
                      self.weight_dict["loss_dice"] * loss_dice +
                      self.weight_dict["loss_iou"] * loss_iou)

        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_iou": loss_iou,
            "loss_total": total_loss,
        }