from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import List, Dict, Any, Optional, Callable, Union
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

# --------------------- Utilities ---------------------

def _to_bool_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Accepts HxW or [N,H,W] arrays/tensors, binarizes (>0) and returns bool tensor on device with shape [N,H,W].
    """
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = arr[None, ...]
        t = torch.from_numpy(arr)
    elif isinstance(x, torch.Tensor):
        t = x
        if t.ndim == 2:
            t = t.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported mask type: {type(x)}")
    # binarize and cast to bool
    t = (t != 0)
    return t.to(device=device, dtype=torch.bool)


def _downsample_bool_masks(m: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample boolean masks by a small integer factor via max-pooling (keeps foreground coverage).
    m: [N,H,W] bool
    """
    if factor <= 1 or m.numel() == 0:
        return m
    # reshape for pooling
    N, H, W = m.shape
    H2 = H // factor
    W2 = W // factor
    if H2 == 0 or W2 == 0:
        return m
    # crop to divisible
    m = m[:, :H2 * factor, :W2 * factor]
    # convert to float for pooling-like reduction via unfold
    mf = m.float()
    mf = mf.unfold(1, factor, factor).unfold(2, factor, factor)  # [N, H2, W2, f, f]
    # max over the small window -> any(True)
    mf = mf.contiguous().view(N, H2, W2, -1).max(dim=-1).values
    return (mf > 0).to(dtype=torch.bool)


# --------------------- IoU (vectorized) ---------------------

def _pairwise_iou_bool(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    a: [Na,H,W] bool, b: [Nb,H,W] bool -> IoU [Na,Nb] float32
    Vectorized via flatten + matmul. Stays on device.
    """
    Na = a.shape[0]
    Nb = b.shape[0]
    if Na == 0 or Nb == 0:
        return a.new_zeros((Na, Nb), dtype=torch.float32)

    # Flatten (reshape tolerates non-contiguous inputs)
    a_f = a.reshape(Na, -1).float()
    b_f = b.reshape(Nb, -1).float()

    # Areas and intersections
    areas_a = a_f.sum(dim=1)             # [Na]
    areas_b = b_f.sum(dim=1)             # [Nb]
    inter = a_f @ b_f.t()                # [Na,Nb]

    # Unions
    union = areas_a[:, None] + areas_b[None, :] - inter + eps
    return (inter / union).to(torch.float32)


# --------------------- Metrics ---------------------

def abiou_original(proposals: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    """
    ABIoU as mean over GTs of max IoU to any proposal (allows proposal reuse).
    proposals, gts: [N,H,W] bool (on same device)
    """
    dev = gts.device if gts.numel() > 0 else (proposals.device if proposals.numel() > 0 else torch.device("cpu"))
    if gts.numel() == 0 and proposals.numel() == 0:
        return torch.tensor(1.0, device=dev, dtype=torch.float32)
    if gts.numel() == 0 or proposals.numel() == 0:
        return torch.tensor(0.0, device=dev, dtype=torch.float32)

    iou = _pairwise_iou_bool(gts, proposals)  # [Ng,Np]
    best = iou.max(dim=1).values              # [Ng]
    return best.mean()


def abiou_one_to_one_greedy(proposals: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    """
    ABIoU with one-to-one greedy matching (no proposal reuse).
    proposals, gts: [N,H,W] bool
    """
    dev = gts.device if gts.numel() > 0 else (proposals.device if proposals.numel() > 0 else torch.device("cpu"))
    if gts.numel() == 0 and proposals.numel() == 0:
        return torch.tensor(1.0, device=dev, dtype=torch.float32)
    if gts.numel() == 0 or proposals.numel() == 0:
        return torch.tensor(0.0, device=dev, dtype=torch.float32)

    iou = _pairwise_iou_bool(gts, proposals)  # [Ng,Np]
    Ng, Np = iou.shape
    used_g = torch.zeros(Ng, dtype=torch.bool, device=dev)
    used_p = torch.zeros(Np, dtype=torch.bool, device=dev)

    matched_sum = torch.tensor(0.0, device=dev)
    # Greedy loop at most min(Ng, Np) steps
    for _ in range(min(Ng, Np)):
        # mask used rows/cols by setting to -1
        iou_masked = iou.clone()
        if used_g.any():
            iou_masked[used_g, :] = -1
        if used_p.any():
            iou_masked[:, used_p] = -1
        val, idx = torch.max(iou_masked.view(-1), dim=0)
        if val <= 0:
            break
        g_idx = idx // Np
        p_idx = idx % Np
        matched_sum = matched_sum + val
        used_g[g_idx] = True
        used_p[p_idx] = True

    # Average over ALL GTs (unmatched GTs count 0)
    return matched_sum / max(Ng, 1)


def union_iou(proposals: torch.Tensor, gts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pixel-set IoU between union of proposals and union of GTs.
    proposals, gts: [N,H,W] bool
    """
    dev = gts.device if gts.numel() > 0 else (proposals.device if proposals.numel() > 0 else torch.device("cpu"))
    if gts.numel() == 0 and proposals.numel() == 0:
        return torch.tensor(1.0, device=dev, dtype=torch.float32)

    if proposals.numel() > 0:
        P = proposals.any(dim=0)
    else:
        # create zero map based on gts ref
        H, W = gts.shape[-2], gts.shape[-1]
        P = torch.zeros((H, W), dtype=torch.bool, device=dev)

    if gts.numel() > 0:
        G = gts.any(dim=0)
    else:
        H, W = proposals.shape[-2], proposals.shape[-1]
        G = torch.zeros((H, W), dtype=torch.bool, device=dev)

    inter = (P & G).sum().float()
    uni = (P | G).sum().float() + eps
    return (inter / uni).to(torch.float32)


# --------------------- Main evaluator ---------------------

@torch.no_grad()
def automask_metrics(
    sam2_model_or_predictor: Any,
    images: List[Union[np.ndarray, torch.Tensor]],   # HxW or HxWx3 (uint8 preferred)
    gt_masks_per_image: List[List[Union[np.ndarray, torch.Tensor]]],  # per-image list of HxW masks
    *,
    amg_kwargs: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 200,
    device: Optional[torch.device] = None,
    autocast_ctx: Optional[Callable[[], Any]] = None,
    downsample_factor: int = 1,
    return_per_image: bool = False,
) -> Dict[str, Any]:
    """
    Run SAM2AutomaticMaskGenerator once per image and compute:
      - ABIoU_one_to_one (greedy, no reuse)
      - UnionIoU
      - ABIoU_original (optional reference)

    Speed features:
      - Single IoU matrix fuels both ABIoUs.
      - Everything stays on GPU; masks are boolean.
      - Optional downsample_factor (e.g., 2 or 4) for huge speedups.

    Returns:
      {
        'ABIoU_one_to_one': float,
        'UnionIoU': float,
        'ABIoU_original': float,
        'num_images': int,
        'per_image': [ ... ]  # if return_per_image
      }
    """

    # AMG defaults (safe, tweak as needed)
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

    model_for_amg = getattr(sam2_model_or_predictor, "model", sam2_model_or_predictor)
    mask_generator = SAM2AutomaticMaskGenerator(model=model_for_amg, **_amg)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autocast (AMG forward only)
    ac = autocast_ctx if autocast_ctx is not None else (lambda: nullcontext())

    # Accumulators
    one2one_vals, union_vals, abiou_orig_vals = [], [], []
    per_image_out = []

    for img, gt_list in zip(images, gt_masks_per_image):
        # ---- Ensure numpy uint8 image for AMG ----
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img
        H, W = img_np.shape[:2]

        # ---- AMG forward ----
        with ac():
            proposals = mask_generator.generate(img_np)  # list of dict

        # ---- Convert proposals -> [Np,H,W] bool on device ----
        if len(proposals) == 0:
            prop_masks = torch.zeros((0, H, W), dtype=torch.bool, device=device)
        else:
            # sort by predicted_iou (or score), keep top_k
            def _score(d):
                return float(d.get("predicted_iou", d.get("score", 0.0)))
            proposals.sort(key=_score, reverse=True)
            if top_k is not None and top_k > 0:
                proposals = proposals[:top_k]
            masks_np = [(p["segmentation"] > 0).astype(np.uint8) for p in proposals]
            prop_masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(device=device, dtype=torch.bool)
            prop_masks = prop_masks.contiguous()

        # ---- Convert GTs -> [Ng,H,W] bool on device ----
        if len(gt_list) == 0:
            gt_masks = torch.zeros((0, H, W), dtype=torch.bool, device=device)
        else:
            gt_bool = []
            for g in gt_list:
                if isinstance(g, torch.Tensor):
                    g_np = g.detach().cpu().numpy()
                else:
                    g_np = g
                gt_bool.append((g_np > 0).astype(np.uint8))
            gt_masks = torch.from_numpy(np.stack(gt_bool, axis=0)).to(device=device, dtype=torch.bool)
            gt_masks = gt_masks.contiguous()

        # ---- Optional downsample (max-pool style) ----
        if downsample_factor > 1:
            prop_masks_ds = _downsample_bool_masks(prop_masks, downsample_factor)
            gt_masks_ds = _downsample_bool_masks(gt_masks, downsample_factor)
        else:
            prop_masks_ds = prop_masks
            gt_masks_ds = gt_masks

        # ---- Metrics (single IoU matrix shared under the hood) ----
        m_one2one = abiou_one_to_one_greedy(prop_masks_ds, gt_masks_ds)
        m_union = union_iou(prop_masks_ds, gt_masks_ds)
        m_orig = abiou_original(prop_masks_ds, gt_masks_ds)

        one2one_vals.append(m_one2one)
        union_vals.append(m_union)
        abiou_orig_vals.append(m_orig)

        if return_per_image:
            per_image_out.append({
                "ABIoU": float(m_one2one.detach().cpu()),
                "ABIoU_original": float(m_orig.detach().cpu()),
                "num_props": int(prop_masks.shape[0]),
                "num_gt": int(gt_masks.shape[0]),
                "H": int(H),
                "W": int(W),
            })

    # ---- Averages ----
    if len(one2one_vals) == 0:
        return {
            "ABIoU_one_to_one": 0.0,
            "UnionIoU": 0.0,
            "ABIoU_original": 0.0,
            "num_images": 0,
            "per_image": [],
        }

    ABIoU_one_to_one = torch.stack(one2one_vals).mean().item()
    ABIoU_original_avg = torch.stack(abiou_orig_vals).mean().item()

    out = {
        "ABIoU": ABIoU_one_to_one,
        "ABIoU_original": ABIoU_original_avg,
        "num_images": len(one2one_vals),
    }
    if return_per_image:
        out["per_image"] = per_image_out
    return out
