from saber.visualization.classifier import get_colors, add_masks
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2, torch, csv, os
import numpy as np

def mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    """xyxy box from a binary mask (H,W) in {0,1}."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def sample_positive_points(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Sample k (x,y) positives uniformly from mask pixels."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = np.random.randint(0, xs.size, size=k)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

def instances_from_grid(grid_points: np.ndarray, segmentation: np.ndarray) -> list[int]:
    """Return unique non-zero instance IDs at the given (x,y) grid sample points."""
    h, w = segmentation.shape
    xs = np.clip(grid_points[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(grid_points[:, 1].astype(np.int32), 0, h - 1)
    ids = segmentation[ys, xs]
    ids = ids[ids > 0]
    return np.unique(ids).astype(int).tolist()

# helper: split an instance id into connected components
def components_for_id(seg, iid: int, min_pixels: int):
    mask_u8 = (seg == iid).astype(np.uint8)        # 0/1
    num, lbl = cv2.connectedComponents(mask_u8)
    comps = []
    for cid in range(1, num):                      # skip background=0
        comp = (lbl == cid).astype(np.float32)     # HxW {0,1}
        if comp.sum() >= min_pixels:
            comps.append(comp)
    return comps


def collate_autoseg(batch, max_res: int = 1024):
    """
    Aspect-preserving resize to fit within max_res, then pad each sample
    to the batch's (H_max, W_max). Keep ragged structures (points/labels/boxes)
    as lists per instance.
    """
    processed = [_resize_one(sample, max_res) for sample in batch]

    # Common padded size for this batch
    H_max = max(s["image"].shape[0] for s in processed)
    W_max = max(s["image"].shape[1] for s in processed)

    for s in processed:
        h, w = s["image"].shape[:2]

        # pad image (top-left anchoring)
        pad_img = np.zeros((H_max, W_max, 3), dtype=s["image"].dtype)
        pad_img[:h, :w] = s["image"]
        s["image"] = pad_img

        # pad masks
        padded_masks = []
        for m in s["masks"]:
            pm = np.zeros((H_max, W_max), dtype=np.uint8)
            pm[:h, :w] = m
            padded_masks.append(pm)
        s["masks"] = padded_masks

        # NOTE: points/boxes coords don't change with top-left padding

    # Return exactly what your trainer expects: lists of per-sample items,
    # and for ragged things, lists of per-instance tensors.
    return {
        "images": [s["image"] for s in processed],  # list of HxWx3 uint8 (predictor handles numpy)
        "masks":  [torch.from_numpy(np.stack(s["masks"])) for s in processed],  # list of [M,H,W]
        "points": [ [torch.from_numpy(p) for p in s["points"]] for s in processed],  # list of list[[Pi,2]]
        "labels": [ [torch.as_tensor(l) for l in s["labels"]] for s in processed],   # list of list[[Pi]]
        "boxes":  [ [torch.from_numpy(b) for b in s["boxes"]] for s in processed],   # list of list[[4]]
    }

def _resize_one(s, max_res: int):
    """
    Resize one sample with aspect preserved. Scale coords per instance.
    Keeps ragged structures as lists.
    """
    img    = s["image"]
    masks  = s["masks"]     # list of [H,W]
    points = s["points"]    # list of [Pi,2]
    labels = s["labels"]    # list of [Pi]
    boxes  = s["boxes"]     # list of [4]

    H, W = img.shape[:2]
    r = min(max_res / max(H, W), 1.0)  # cap longest side; don't upscale
    newH, newW = int(round(H * r)), int(round(W * r))

    if (newH, newW) != (H, W):
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        masks = [cv2.resize(m.astype(np.uint8), (newW, newH), interpolation=cv2.INTER_NEAREST) for m in masks]
        pts   = [np.asarray(p, dtype=np.float32) * r for p in points]  # scale each instance
        bxs   = [np.asarray(b, dtype=np.float32) * r for b in boxes]
    else:
        pts = [np.asarray(p, dtype=np.float32) for p in points]
        bxs = [np.asarray(b, dtype=np.float32) for b in boxes]

    # labels stay as-is per instance
    lbls = [np.asarray(l) for l in labels]

    return {
        "image": img,
        "masks": masks,
        "points": pts,
        "labels": lbls,
        "boxes":  bxs,
    }

def _to_numpy_mask_stack(masks):
    """
    Accepts list/tuple of tensors or np arrays shaped [H,W];
    returns np.uint8 array [N,H,W] with values {0,1}.
    """
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        arr = masks
    else:
        arr = np.stack([m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
                        for m in masks], axis=0)
    # binarize & cast
    arr = (arr > 0).astype(np.uint8)
    return arr

def visualize_item_with_points(image, masks, points, boxes=None,
                               title=None, point_size=24):
    """
    Show a single image with all component masks (colored),
    all positive points (color-matched to masks), and optional boxes.

    Args
    ----
    image : np.ndarray  (H,W) or (H,W,3)  (uint8 or float)
    masks : list[np.ndarray or torch.Tensor] each [H,W] in {0,1}
    points: list[np.ndarray] each [P,2] as (x,y)
    boxes : list[np.ndarray] each [4] xyxy (optional)
    """
    # normalize inputs
    mstack = _to_numpy_mask_stack(masks)      # [N,H,W] in {0,1}

    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(image, interpolation="nearest")

    # overlay masks with your helper
    add_masks(mstack, ax)  # uses get_colors internally

    # color-match points (and boxes) to mask color
    colors = get_colors(len(masks))
    for i, pts in enumerate(points):
        if pts is None or len(pts) == 0:
            continue
        color = colors[i % len(colors)]
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size,
                   c=[(color[0], color[1], color[2], 1.0)],
                   edgecolors="k", linewidths=0.5)

    if boxes is not None:
        for i, b in enumerate(boxes):
            if b is None: 
                continue
            color = colors[i % len(colors)]
            x0, y0, x1, y1 = map(float, b)
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2,
                edgecolor=(color[0], color[1], color[2], 1.0),
                facecolor="none"
            )
            ax.add_patch(rect)

    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

def save_training_log(results, outdir="results"):

    # CSV (epoch-aligned, pad with blanks if needed)
    path = os.path.join(outdir, "metrics.csv")
    is_new = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "val_loss", "ABIoU"])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "epoch": int(results['epoch']),
            "lr": f"{results['lr']:.1e}",
            "train_loss": float(results['train']['loss']),
            "val_loss": float(results['loss']),
            "ABIoU": float(results['ABIoU']),
        })

########################################################################################

def _orig_hw_tuple(pred):
    ohw = pred._orig_hw
    # Cases seen in SAM/SAM2 wrappers: (H,W), [H,W], [(H,W)], possibly numpy ints
    if isinstance(ohw, list):
        # list of tuples -> pick the last tuple
        if len(ohw) and isinstance(ohw[-1], (list, tuple)):
            h, w = ohw[-1]
        else:
            # flat [H, W]
            h, w = ohw[0], ohw[1]
    elif isinstance(ohw, tuple):
        h, w = ohw
    else:
        # e.g., numpy array-like
        h, w = ohw[0], ohw[1]
    return (int(h), int(w))

@torch.no_grad()
def infer_on_single_image(
    predictor,
    image,
    inst_points,     # list[Tensor(Pi,2)]
    inst_labels,     # list[Tensor(Pi)]
    inst_masks=None, # optional list[H,W]
    inst_boxes=None, # list[Tensor(4)] or None
    use_boxes=True,
    predict_multimask=False,
    device="cuda",
):
    # 1) Encode image once
    predictor.set_image(image)

    # 2) Normalize cached features to [1,C,H',W']
    def _to_batched_4d(x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 3: x = x.unsqueeze(0)
        return x.to(device)

    image_embeddings = _to_batched_4d(predictor._features["image_embed"])               # [1,C,H′,W′]
    high_res_features = [_to_batched_4d(lvl) for lvl in predictor._features["high_res_feats"]]

    # 3) Pack prompts to (N,P,2) / (N,P)
    pts_all  = [torch.as_tensor(p, device=device, dtype=torch.float32) for p in inst_points]
    lbl_all  = [torch.as_tensor(l, device=device, dtype=torch.float32) for l in inst_labels]
    N = len(pts_all)
    if N == 0:
        return None, None, None

    P = max(p.shape[0] for p in pts_all)
    pts_pad = torch.zeros((N, P, 2), device=device)
    lbl_pad = torch.full((N, P), -1.0, device=device)
    for i,(p,l) in enumerate(zip(pts_all, lbl_all)):
        pts_pad[i, :p.shape[0]] = p
        lbl_pad[i, :l.shape[0]] = l

    boxes = None
    if use_boxes and inst_boxes:
        boxes = torch.stack([torch.as_tensor(bx, device=device, dtype=torch.float32) for bx in inst_boxes], dim=0)

    # 4) Prompt encoding
    _, unnorm_coords, labels, _ = predictor._prep_prompts(
        pts_pad, lbl_pad, box=boxes, mask_logits=None, normalize_coords=True
    )
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=boxes if use_boxes else None,
        masks=None,
    )

    # 5) Decode with repeat_image=True  ✅ no manual feature tiling
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=image_embeddings,   # [1,C,H′,W′]
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,   # [N, ...]
        dense_prompt_embeddings=dense_embeddings,     # [N, ...]
        multimask_output=predict_multimask,
        repeat_image=True,                  # <-- let decoder repeat internally
        high_res_features=high_res_features,          # each [1,C,H′,W′]
    )

    # 6) Upscale + optional stack GT
    def _orig_hw_tuple(pred):
        ohw = pred._orig_hw
        if isinstance(ohw, list) and len(ohw) and isinstance(ohw[-1], (list, tuple)):
            return (int(ohw[-1][0]), int(ohw[-1][1]))
        if isinstance(ohw, tuple):
            return (int(ohw[0]), int(ohw[1]))
        return tuple(int(v) for v in ohw)

    out_hw = _orig_hw_tuple(predictor)  # (H, W) e.g., (928, 960)
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, out_hw)  # [N,K,H,W]

    gt_masks = None
    if inst_masks:
        gt_masks = torch.stack([torch.as_tensor(m, device=device).float() for m in inst_masks], dim=0)

    return prd_masks, prd_scores, gt_masks