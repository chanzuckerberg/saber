from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
import numpy as np
import torch

# Subset of IoU thresholds, as requested:
AR_THRESHOLDS = np.array([0.50, 0.65, 0.75, 0.85], dtype=np.float32)

# ------------------------ Decoder-side helpers ------------------------

def _binary_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fast IoU for binary masks (boolean or {0,1} tensors).
    Shapes: a,b: (H, W)
    """
    inter = (a & b).float().sum()
    uni   = (a | b).float().sum().clamp_min(eps)
    return inter / uni

@torch.no_grad()
def decoder_prompt_miou(prd_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """
    Best-of-K decoder prompt mIoU.
    Args:
        prd_masks: [N, K, H, W] LOGITS from the decoder
        gt_masks:  [N, H, W]    binary {0,1}
    Returns:
        mean over N of (max IoU over K), using SAM2's thresholding rule (logits > 0).
    """
    N, K, H, W = prd_masks.shape
    # SAM2 convention: threshold decoder outputs by logits > 0
    pred_bin = (prd_masks > 0)  # bool, [N,K,H,W]
    ious = []
    for n in range(N):
        gt = gt_masks[n].bool()
        if gt.sum() == 0:
            continue  # skip empty GT
        best = torch.stack([_binary_iou(pred_bin[n, k], gt) for k in range(K)], dim=0).max()
        ious.append(best)
    if len(ious) == 0:
        return float("nan")
    return float(torch.stack(ious).mean().item())

@torch.no_grad()
def iou_head_calibration_from_decoder(
    prd_masks: torch.Tensor,
    prd_scores: torch.Tensor,
    gt_masks:   torch.Tensor,
    num_bins:   int = 15,
) -> Dict[str, Any]:
    """
    Compare predicted IoU (sigmoid(prd_scores)) vs true IoU (from logits>0 masks).
    Args:
        prd_masks:  [N,K,H,W] LOGITS
        prd_scores: [N,K]     raw IoU logits
        gt_masks:   [N,H,W]
    Returns:
        dict with calibration MAE, Brier, ECE, and a per-bin table (for diagnostics).
    """
    N, K, H, W = prd_masks.shape
    preds, trues = [], []
    pred_bin = (prd_masks > 0)  # bool

    for n in range(N):
        gt = gt_masks[n].bool()
        if gt.sum() == 0:
            continue
        # True IoU per K proposal
        true_iou_k = torch.stack([_binary_iou(pred_bin[n, k], gt) for k in range(K)], dim=0)  # [K]
        pred_iou_k = prd_scores[n].sigmoid().clamp(0, 1)                                       # [K]
        trues.append(true_iou_k)
        preds.append(pred_iou_k)

    if len(preds) == 0:
        return {"mae": float("nan"), "brier": float("nan"), "ece": float("nan"), "table": []}

    preds = torch.cat(preds)  # [N'*K]
    trues = torch.cat(trues)  # [N'*K]

    mae   = torch.abs(preds - trues).mean().item()
    brier = torch.mean((preds - trues) ** 2).item()

    # Expected Calibration Error (ECE) over uniform bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    pred_np, true_np = preds.cpu().numpy(), trues.cpu().numpy()
    idx = np.clip(np.digitize(pred_np, bins, right=True) - 1, 0, num_bins - 1)

    ece = 0.0
    table = []
    total = len(pred_np)
    for b in range(num_bins):
        m = (idx == b)
        n_b = int(m.sum())
        if n_b == 0:
            table.append({"bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})", "count": 0,
                          "mean_pred": None, "mean_true": None, "gap": None})
            continue
        mp = float(pred_np[m].mean())
        mt = float(true_np[m].mean())
        gap = abs(mp - mt)
        ece += (n_b / total) * gap
        table.append({"bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})", "count": n_b,
                      "mean_pred": round(mp, 4), "mean_true": round(mt, 4), "gap": round(gap, 4)})

    return {"mae": mae, "brier": float(brier), "ece": float(ece), "table": table}

# ------------------------ AMG proposal metrics ------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean numpy masks (H,W)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(1.0, float(union))

def _iou_matrix(preds: List[np.ndarray], gts: List[np.ndarray]) -> np.ndarray:
    """
    Build P x G IoU matrix for numpy boolean masks.
    preds: list of predicted masks (H,W)
    gts:   list of gt masks      (H,W)
    """
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)
    M = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            M[i, j] = _iou(p, g)
    return M

def _greedy_match(M: np.ndarray, tau: float) -> Tuple[int, int, int]:
    """
    Greedy bipartite matching by IoU descending with threshold tau.
    Returns:
        TP, FP, FN
    """
    P, G = M.shape
    used_p, used_g, matches = set(), set(), []
    pairs = [(i, j, M[i, j]) for i in range(P) for j in range(G)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, iou in pairs:
        if iou < tau:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matches.append((i, j))
    TP = len(matches); FP = P - TP; FN = G - TP
    return TP, FP, FN

def average_recall_amg(
    amg_outputs: List[List[dict]],
    gt_masks_list: List[List[torch.Tensor]],
    iou_thresholds: np.ndarray = AR_THRESHOLDS,
    max_proposals: int = None,
) -> Dict[str, Any]:
    """
    Average Recall across IoU thresholds.
    Args:
        amg_outputs:   list over images of list[mask_dict]; each dict has 'segmentation' (np.bool array) and 'predicted_iou'
        gt_masks_list: list over images of list[tensor HxW] for ground-truth instances
        iou_thresholds: numpy array of IoU taus to average over
        max_proposals: cap #proposals per image (after ranking by predicted_iou) for speed
    Returns:
        {"AR": scalar, "per_tau_recall": {tau: recall_tau}}
    """
    recalls = []
    for tau in iou_thresholds:
        tp = fn = 0
        for img_masks, gts in zip(amg_outputs, gt_masks_list):
            # rank by predicted_iou then cap
            if max_proposals is not None:
                img_masks = sorted(img_masks, key=lambda d: d.get('predicted_iou', 0.0), reverse=True)[:max_proposals]
            preds = [m['segmentation'].astype(bool) for m in img_masks]
            gts_np = [g.cpu().numpy().astype(bool) for g in gts]
            M = _iou_matrix(preds, gts_np)
            tpi, _, fni = _greedy_match(M, float(tau))
            tp += tpi; fn += fni
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else np.nan)

    per_tau = {float(t): (None if np.isnan(r) else float(r)) for t, r in zip(iou_thresholds, recalls)}
    return {"AR": float(np.nanmean(recalls)), "per_tau_recall": per_tau}

def recall_at_k_amg(
    amg_outputs: List[List[dict]],
    gt_masks_list: List[List[torch.Tensor]],
    ks: Tuple[int, ...] = (10, 50, 100),
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Recall@K at a fixed IoU threshold (default 0.5).
    """
    out = {}
    for K in ks:
        tp = fn = 0
        for img_masks, gts in zip(amg_outputs, gt_masks_list):
            sel = sorted(img_masks, key=lambda d: d.get('predicted_iou', 0.0), reverse=True)[:K]
            preds = [m['segmentation'].astype(bool) for m in sel]
            gts_np = [g.cpu().numpy().astype(bool) for g in gts]
            M = _iou_matrix(preds, gts_np)
            tpi, _, fni = _greedy_match(M, iou_thresh)
            tp += tpi; fn += fni
        denom = tp + fn
        out[K] = tp / denom if denom > 0 else float('nan')
    return {"Recall@K": out, "iou_thresh": iou_thresh}

# ------------------------ Wrapper for validation loop ------------------------

@torch.no_grad()
def sam2_metrics(batch: Dict[str, Any], outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], amg) -> Dict[str, Any]:
    """
    Compute a SAM2-style metric bundle for a single validation batch.
    Args:
        batch:   {"images": list[np.ndarray or torch tensor], "masks": list[list[torch.Tensor HxW]]}
        outputs: (prd_masks, prd_scores, gt_masks) where
                 prd_masks: [N,K,H,W] logits, prd_scores: [N,K] logits, gt_masks: [N,H,W] {0,1}
        amg:     a SAM2AutomaticMaskGenerator instance
    Returns:
        dict with prompt_mIoU, IoU calibration (mae/brier/ece + table),
        AR averaged over {0.50,0.65,0.75,0.85}, per-threshold recalls,
        and Recall@{10,50,100}. Includes "num_images" for weighted averaging.
    """
    prd_masks, prd_scores, gt_masks = outputs[:3]

    # Decoder-side metrics (cheap; no extra forward)
    pm = decoder_prompt_miou(prd_masks, gt_masks)
    cal = iou_head_calibration_from_decoder(prd_masks, prd_scores, gt_masks, num_bins=15)

    # AMG proposals (dominates runtime; run once, reuse for AR and R@K)
    all_amg = [amg.generate(img) for img in batch["images"]]
    all_gt  = batch["masks"]

    ar = average_recall_amg(all_amg, all_gt, iou_thresholds=AR_THRESHOLDS, max_proposals=200)
    rK = recall_at_k_amg(all_amg, all_gt, ks=(10, 50, 100), iou_thresh=0.5)

    return {
        "prompt_miou": float(pm) if pm == pm else float("nan"),
        "cal_mae":   cal["mae"],
        "cal_brier": cal["brier"],
        "cal_ece":   cal["ece"],
        "cal_tables": cal["table"],             # keep for inspection; don’t reduce across ranks
        "AR": ar["AR"],
        "num_images": len(batch["images"]),
        "R@10": rK["Recall@K"][10],
        "R@50": rK["Recall@K"][50],
        "R@100": rK["Recall@K"][100],
    }
        # # "per_tau_recall": ar["per_tau_recall"], # keep for plotting; don’t reduce as a scalar
        # "R@10": rK["Recall@K"][10],
        # "R@50": rK["Recall@K"][50],