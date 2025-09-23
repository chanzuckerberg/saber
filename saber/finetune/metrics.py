from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from contextlib import nullcontext
import numpy as np
import torch

# --------------------- IoU / ABIoU ---------------------

def _mask_iou(a, b, eps=1e-6):
    """
    a: [Na,H,W] {0,1}; b: [Nb,H,W] {0,1} -> IoU [Na,Nb]
    """
    if a.numel() == 0 or b.numel() == 0:
        dev = a.device if a.numel() > 0 else (b.device if b.numel() > 0 else torch.device("cpu"))
        Na = a.shape[0] if a.numel() > 0 else 0
        Nb = b.shape[0] if b.numel() > 0 else 0
        return torch.zeros((Na, Nb), device=dev, dtype=torch.float32)

    a = a.float()
    b = b.float()
    inter = torch.einsum("nhw,mhw->nm", a, b)   # [Na,Nb]

    ua = a.sum(dim=(1,2))[:, None]              # [Na,1]
    ub = b.sum(dim=(1,2))[None, :]              # [1,Nb]

    union = ua + ub - inter + eps               # [Na,Nb]
    return inter / union

def _abiou(proposals, gts):
    """
    Average Best IoU (coverage metric).
    proposals: [Np,H,W] {0,1}, gts: [Ng,H,W] {0,1}
    """
    if gts.numel() == 0 and proposals.numel() == 0:
        dev = proposals.device if proposals.numel() > 0 else gts.device
        return torch.tensor(1.0, device=dev, dtype=torch.float32)
    if gts.numel() == 0 or proposals.numel() == 0:
        dev = proposals.device if proposals.numel() > 0 else gts.device
        return torch.tensor(0.0, device=dev, dtype=torch.float32)
    iou = _mask_iou(gts, proposals)          # [Ng,Np]
    best = iou.max(dim=1).values             # [Ng]
    return best.mean()

# --------------------- ABIoU evaluator ---------------------

@torch.no_grad()
def automask_metrics(
    sam2_model_or_predictor,
    images,
    gt_masks_per_image,
    *,
    amg_kwargs=None,
    top_k=20,
    device=None,
    autocast_ctx=None,   # callable -> context manager (e.g., trainer.autocast); if None, no autocast
):
    """
    Run SAM2AutomaticMaskGenerator and compute ABIoU only.

    Args
    ----
    sam2_model_or_predictor : SAM2 model or predictor (we’ll use .model if present)
    images : list of images: HxW or HxWx3; uint8 NumPy preferred, but float is fine
    gt_masks_per_image : list[list[H x W]]; elements may be NumPy or torch; non-zero = foreground
    amg_kwargs : dict forwarded to SAM2AutomaticMaskGenerator
    top_k : keep only top-K proposals per image by AMG score (optional)
    device : torch device for tensors (defaults to 'cuda' if available)
    autocast_ctx : callable returning a context manager for mixed precision during AMG forward only
    """
    # Defaults for AMG (tweak as you like)
    _amg = dict(
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.7,
        use_m2m=False,
        multimask_output=True,
    )
    if amg_kwargs:
        _amg.update(amg_kwargs)
    print(_amg)
    model_for_amg = getattr(sam2_model_or_predictor, "model", sam2_model_or_predictor)
    mask_generator = SAM2AutomaticMaskGenerator(model=model_for_amg, **_amg)

    # Determine Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AutoCast 
    ac = autocast_ctx if autocast_ctx is not None else (lambda: nullcontext())

    print('TEST')

    # Per-image loop
    per_image, abiou_vals = [], []
    for img, gt_list in zip(images, gt_masks_per_image):
        # -------- ensure AMG-friendly uint8 numpy image --------
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img
        H, W = img_np.shape[:2]

        # -------- AMG forward under autocast (fast path) --------
        with ac():
            props = mask_generator.generate(img_np)  # list of dicts

        if len(props) == 0:
            prop_masks = torch.zeros((0, H, W), dtype=torch.uint8, device=device)
        else:
            # Sort by predicted_iou/score and optionally keep top_k
            def _score(d):
                return float(d.get("predicted_iou", d.get("score", 0.0)))
            props.sort(key=_score, reverse=True)
            if top_k is not None and top_k > 0:
                props = props[:top_k]

            masks_np = [(p["segmentation"] > 0).astype(np.uint8) for p in props]
            prop_masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(device=device)

        # -------- stack GTs --------
        if len(gt_list) == 0:
            gt_masks = torch.zeros((0, H, W), dtype=torch.uint8, device=device)
        else:
            gts = []
            for g in gt_list:
                if isinstance(g, torch.Tensor):
                    g_np = g.detach().cpu().numpy()
                else:
                    g_np = g
                gts.append((g_np > 0).astype(np.uint8))
            gt_masks = torch.from_numpy(np.stack(gts, axis=0)).to(device=device)

        # -------- ABIoU in float32 (stable) --------
        abiou = _abiou(prop_masks, gt_masks)
        per_image.append({
            "ABIoU": float(abiou.detach().cpu()),
            "num_props": int(prop_masks.shape[0]),
            "num_gt": int(gt_masks.shape[0]),
        })
        abiou_vals.append(abiou)

    ABIoU = torch.stack(abiou_vals).mean().item() if abiou_vals else 0.0
    return {
        "ABIoU": ABIoU,
        'ABIoU_per_image': abiou_vals,
        "num_images": len(per_image),
        "per_image": per_image,
    }
