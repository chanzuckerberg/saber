import torch.nn.functional as F
import torch.nn as nn
import torch

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return 1 - (2 * inter + eps) / (denom + eps)

def focal_loss_from_logits(logits, targets, alpha=0.25, gamma=2.0):
    # logits, targets: (N,H,W) float with targets in {0,1}
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (w * ((1 - pt) ** gamma) * ce).mean(dim=(1, 2))  # -> (N,)

class MultiMaskIoULoss(nn.Module):
    def __init__(
        self,
        iou_regression: str = "l1",              # "l1" or "mse"
        supervise_all_iou: bool = False,
        all_iou_weight: float = 0.1,
        # Weights: match the (dice=20, focal=1, iou=1, class=1) convention
        weight_dict: dict = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Objectness head
        pred_obj_scores: bool = True,
        focal_alpha_obj: float = 0.5,            # can be -1 to disable alpha weighting
        focal_gamma_obj: float = 0.0,            # 0 -> plain BCE on objectness
        gate_by_objectness: bool = True,         # gate seg/IoU losses when no object
    ):
        super().__init__()
        if weight_dict is None:
            weight_dict = {"loss_mask": 1.0, "loss_dice": 20.0, "loss_iou": 1.0, "loss_class": 1.0}
        # NOTE: "loss_mask" == focal; "loss_dice" == dice; "loss_class" == objectness
        self.focal_weight = weight_dict.get("loss_mask", 1.0)
        self.dice_weight = weight_dict.get("loss_dice", 20.0)
        self.iou_head_weight = weight_dict.get("loss_iou", 1.0)
        self.class_weight = weight_dict.get("loss_class", 1.0)

        self.iou_regression = iou_regression
        self.supervise_all_iou = supervise_all_iou
        self.all_iou_weight = all_iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.pred_obj_scores = pred_obj_scores
        self.focal_alpha_obj = focal_alpha_obj
        self.focal_gamma_obj = focal_gamma_obj
        self.gate_by_objectness = gate_by_objectness

    def _reg_loss(self, pred, target):
        return F.mse_loss(pred, target) if self.iou_regression == "mse" else F.l1_loss(pred, target)

    @staticmethod
    def _binary_iou(bin_mask, gt_bool, eps=1e-6):
        inter = (bin_mask & gt_bool).float().sum(dim=(1, 2))
        union = (bin_mask | gt_bool).float().sum(dim=(1, 2)).clamp_min(eps)
        return inter / union

    def _objectness_targets(self, gt_mask):
        # gt_mask: (H,W) bool/0-1 → scalar 0/1 (any positive pixel?)
        return (gt_mask.sum() > 0).float()

    def _objectness_loss(self, logits_scalar, target_scalar):
        # logits_scalar: shape () or (1,)
        # Use focal-like BCE if gamma>0, else plain BCE
        if self.focal_gamma_obj == 0.0 and self.focal_alpha_obj < 0:
            return F.binary_cross_entropy_with_logits(logits_scalar, target_scalar)
        p = torch.sigmoid(logits_scalar)
        ce = F.binary_cross_entropy_with_logits(logits_scalar, target_scalar, reduction="none")
        pt = p * target_scalar + (1 - p) * (1 - target_scalar)
        if self.focal_alpha_obj >= 0:
            w = self.focal_alpha_obj * target_scalar + (1 - self.focal_alpha_obj) * (1 - target_scalar)
        else:
            w = 1.0
        return (w * ((1 - pt) ** self.focal_gamma_obj) * ce).mean()

    def forward(self, prd_masks_logits, prd_iou_scores, gt_masks, object_score_logits=None):
        """
        prd_masks_logits: (N,K,H,W) or (K,H,W)
        prd_iou_scores:  (N,K)     or (K,)
        gt_masks:        (N,H,W)   or (H,W)
        object_score_logits: (N,) or (N,1) or scalar per instance (optional)
        """
        # ---- normalize shapes to batched form ----
        if prd_masks_logits.dim() == 3:  # (K,H,W) -> (1,K,H,W)
            prd_masks_logits = prd_masks_logits.unsqueeze(0)
        if prd_iou_scores.dim() == 1:    # (K,) -> (1,K)
            prd_iou_scores = prd_iou_scores.unsqueeze(0)
        if gt_masks.dim() == 2:          # (H,W) -> (1,H,W)
            gt_masks = gt_masks.unsqueeze(0)

        N, K, H, W = prd_masks_logits.shape
        gt_bool = gt_masks.bool()                  # (N,H,W)
        gt_float = gt_masks.float()                # (N,H,W)

        # ---- IoU per slot (vectorized) ----
        bin_masks = (prd_masks_logits > 0)         # (N,K,H,W)
        gt_bool_b = gt_bool.unsqueeze(1)           # (N,1,H,W)  <-- key fix
        inter = (bin_masks & gt_bool_b).float().sum(dim=(2,3))                    # (N,K)
        union = (bin_masks | gt_bool_b).float().sum(dim=(2,3)).clamp_min(1e-6)    # (N,K)
        true_iou_per_k = inter / union                                              # (N,K)

        # ---- pick best slot per instance ----
        best_ix = torch.argmax(true_iou_per_k, dim=1)                 # (N,)
        idx_4d = best_ix.view(N,1,1,1).expand(N,1,H,W)
        best_logits = prd_masks_logits.gather(1, idx_4d).squeeze(1)   # (N,H,W)

        # ---- segmentation losses on best slot ----
        dice_per  = dice_loss_from_logits(best_logits, gt_float)      # (N,)
        focal_per = focal_loss_from_logits(
            best_logits, gt_float,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )                                                              # (N,)
        seg_loss = self.dice_weight * dice_per.mean() + self.focal_weight * focal_per.mean()

        # ---- IoU head regression ----
        iou_pred_best   = prd_iou_scores.gather(1, best_ix.view(N,1)).squeeze(1)   # (N,)
        iou_target_best = true_iou_per_k.gather(1, best_ix.view(N,1)).squeeze(1)   # (N,)

        # IMPORTANT: prd_iou_scores are already probs in [0,1]; do NOT apply sigmoid here.
        # Regress directly (L1 or MSE) to the true IoU.
        iou_reg_main = self._reg_loss(iou_pred_best, iou_target_best)

        total = seg_loss + self.iou_head_weight * iou_reg_main

        # ---- optional gentle supervision over all K ----
        iou_reg_all = None
        if self.supervise_all_iou and K > 1 and self.all_iou_weight > 0:
            # SAME: no sigmoid; train all K scores directly toward their per-slot IoUs
            iou_reg_all = self._reg_loss(prd_iou_scores, true_iou_per_k)
            total = total + self.all_iou_weight * iou_reg_all

        # ---- objectness (per instance) ----
        obj_loss = None
        if self.pred_obj_scores:
            assert object_score_logits is not None, "object_score_logits required when pred_obj_scores=True"
            obj_logits = object_score_logits.view(N, -1).mean(dim=1)               # (N,)
            obj_target = (gt_bool.view(N, -1).sum(dim=1) > 0).float()              # (N,)
            obj_loss = self._objectness_loss(obj_logits, obj_target)
            total = total + self.class_weight * obj_loss

            if self.gate_by_objectness:
                has_obj = (obj_target > 0.5).float()                                # (N,)
                gate = has_obj.mean().clamp_min(1e-6)
                total = self.class_weight * obj_loss + gate * (seg_loss + self.iou_head_weight * iou_reg_main)
                if iou_reg_all is not None:
                    total = total + gate * (self.all_iou_weight * iou_reg_all)

        return {
            "loss_total": total,
            "loss_seg": seg_loss.detach(),
            "loss_iou": iou_reg_main.detach(),
            "loss_iou_all": (iou_reg_all.detach() if iou_reg_all is not None else None),
            "loss_class": (obj_loss.detach() if obj_loss is not None else None),
            "loss_mask": focal_per.mean().detach(),
            "loss_dice": dice_per.mean().detach(),
            "true_iou_best": iou_target_best.mean().detach(),
        }