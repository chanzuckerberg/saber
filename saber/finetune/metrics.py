from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch.nn.functional as F
import numpy as np
import torch



# --------------------- IoU / metric helpers ---------------------

def _mask_iou(a, b, eps=1e-6):
    """
    a: [Na,H,W] {0,1}; b: [Nb,H,W] {0,1}
    returns IoU matrix [Na,Nb]
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    a = a.float()
    b = b.float()
    inter = torch.einsum("nhw,mhw->nm", a, b)
    ua = a.sum(dim=(1,2)).unsqueeze(1)
    ub = b.sum(dim=(1,2)).unsqueeze(0)
    union = ua + ub - inter + eps
    return inter / union


def _abiou(proposals, gts):
    """
    Average Best IoU (higher is better).
    proposals: [Np,H,W] {0,1}, gts: [Ng,H,W] {0,1}
    """
    if gts.numel() == 0 and proposals.numel() == 0:
        return torch.tensor(1.0, device=proposals.device if proposals.is_cuda else gts.device)
    if gts.numel() == 0:
        return torch.tensor(0.0, device=proposals.device)
    if proposals.numel() == 0:
        return torch.tensor(0.0, device=gts.device)
    iou = _mask_iou(gts, proposals)         # [Ng,Np]
    best = iou.max(dim=1).values            # [Ng]
    return best.mean()


def _ap_at_threshold(proposals, scores, gts, thr=0.5):
    """
    Greedy one-to-one matching by descending score at a single IoU threshold.
    proposals: [Np,H,W] {0,1}
    scores:    [Np] (higher is better)
    gts:       [Ng,H,W] {0,1}
    """
    # Degenerate cases
    if gts.numel() == 0 and proposals.numel() == 0:
        return torch.tensor(1.0, device=scores.device)
    if proposals.numel() == 0:
        return torch.tensor(0.0, device=scores.device)

    order = scores.argsort(descending=True)
    props = proposals[order]

    matched_gt = torch.zeros((gts.shape[0],), dtype=torch.bool, device=gts.device)
    tp, fp = [], []
    for i in range(props.shape[0]):
        if gts.numel() == 0:
            fp.append(1); tp.append(0); continue
        ious = _mask_iou(props[i:i+1], gts)[0]  # [Ng]
        j = torch.argmax(ious)
        if ious[j] >= thr and not matched_gt[j]:
            matched_gt[j] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    tp = torch.tensor(tp, device=scores.device).cumsum(0)
    fp = torch.tensor(fp, device=scores.device).cumsum(0)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / max(gts.shape[0], 1)

    # Precision envelope + trapezoidal integral over recall
    mrec, idx = torch.sort(recall)
    mpre = precision[idx]
    for k in range(mpre.shape[0] - 2, -1, -1):
        mpre[k] = torch.maximum(mpre[k], mpre[k+1])
    return torch.trapz(mpre, mrec)


def _map(proposals, scores, gts, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)):
    """
    Mean AP across IoU thresholds (COCO-style, class-agnostic).
    """
    aps = [ _ap_at_threshold(proposals, scores, gts, thr=t) for t in thresholds ]
    return torch.stack(aps).mean()


# --------------------- Main evaluator ---------------------

@torch.no_grad()
def automask_metrics(
    sam2_model_or_predictor,
    images,
    gt_masks_per_image,
    *,
    amg_kwargs=None,
    top_k=None,              # optionally cap #proposals/image after AMG (for speed)
    compute_map=True,        # if False, only ABIoU and AP@0.5
    map_thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    device=None,
):
    """
    Run SAM2AutomaticMaskGenerator on each image and compute ABIoU, AP@0.5 (and mAP if requested).

    Args
    ----
    sam2_model_or_predictor : the fine-tuned SAM2 object. If you passed a predictor wrapper,
                              we'll try to hand its `.model` to SAM2AutomaticMaskGenerator.
    images : list of images; each can be:
             - HxWx3 uint8 NumPy (preferred for AMG), or
             - torch.Tensor (H,W) or (H,W,3) float in [0,1] or [0,255]
    gt_masks_per_image : list[list[H x W]]; elements can be NumPy or torch; non-zero = foreground
    amg_kwargs : dict of params forwarded to SAM2AutomaticMaskGenerator(...)
                 e.g. {'points_per_side': 16, 'points_per_batch': 64, 'pred_iou_thresh': 0.7, ...}
    top_k : optional int to keep only top-K proposals per image by score after AMG
    compute_map : whether to compute mAP over multiple IoU thresholds
    map_thresholds : tuple of IoU thresholds
    device : torch device for metrics (defaults to 'cuda' if available)

    Returns
    -------
    summary : dict with aggregated metrics:
        {
          'ABIoU': float,
          'AP50':  float,
          'mAP':   float or None,
          'num_images': int,
          'per_image': [ {'ABIoU':..., 'AP50':..., 'mAP':... or None, 'num_props': int, 'num_gt': int}, ... ]
        }
    """
    # local alias to avoid confusion with user's np
    _amg_kwargs = dict(
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.7,
        use_m2m=False,
        multimask_output=False,
    )
    if amg_kwargs:
        _amg_kwargs.update(amg_kwargs)

    # Figure out what to pass into the generator: underlying model vs predictor
    model_for_amg = getattr(sam2_model_or_predictor, "model", sam2_model_or_predictor)

    mask_generator = SAM2AutomaticMaskGenerator(model=model_for_amg, **_amg_kwargs)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_image = []
    abiou_vals, map_vals = [], []

    for img, gt_list in zip(images, gt_masks_per_image):
        
        # ---------- run AMG ----------
        props = mask_generator.generate(img)  # list of dicts with 'segmentation' and 'predicted_iou'/'score'

        H, W, _ = img.shape
        if len(props) == 0:
            prop_masks = torch.zeros((0, H, W), dtype=torch.uint8, device=device)
            prop_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        else:
            # sort by predicted IoU score, keep top_k if requested
            def _score(d):
                return float(d.get("predicted_iou", d.get("score", 0.0)))
            props = sorted(props, key=_score, reverse=True)
            if top_k is not None and top_k > 0:
                props = props[:top_k]

            masks_np = [(p["segmentation"] > 0).astype(np.uint8) for p in props]
            scores_np = [_score(p) for p in props]
            prop_masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(device=device)
            prop_scores = torch.tensor(scores_np, device=device, dtype=torch.float32)

        # ---------- stack GTs ----------
        if len(gt_list) == 0:
            gt_masks = torch.zeros((0, H, W), dtype=torch.uint8, device=device)
        else:
            gts = []
            for g in gt_list:
                g_np = g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else g
                gts.append((g_np > 0).astype(np.uint8))
            gt_masks = torch.from_numpy(np.stack(gts, axis=0)).to(device=device)

        # ---------- metrics ----------
        abiou = _abiou(prop_masks, gt_masks)
        if compute_map:
            mAP = _map(prop_masks, prop_scores, gt_masks, thresholds=map_thresholds)
        else:
            mAP = None

        per_image.append({
            "ABIoU": float(abiou.detach().cpu()),
            "mAP":   (float(mAP.detach().cpu()) if mAP is not None else None),
            "num_props": int(prop_masks.shape[0]),
            "num_gt": int(gt_masks.shape[0]),
        })

        abiou_vals.append(abiou)
        if compute_map:
            map_vals.append(mAP)

    # aggregate
    ABIoU = torch.stack(abiou_vals).mean().item() if abiou_vals else 0.0
    mAP   = (torch.stack(map_vals).mean().item() if compute_map and map_vals else None)

    return {
        "ABIoU": ABIoU,
        "mAP": mAP,
        "num_images": len(per_image),
        "per_image": per_image,
    }
