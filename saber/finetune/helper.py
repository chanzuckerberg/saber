from saber.visualization.classifier import get_colors, add_masks
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def collate_autoseg(batch):
    # batch: list of dicts from _package_image_item
    return {
        "images": [b["image"] for b in batch],    # list of HxWx3 uint8
        "masks":  [b["masks"] for b in batch],    # list of list[H x W]
        "points": [b["points"] for b in batch],   # list of list[#p x 2]
        "labels": [b["labels"] for b in batch],   # list of list[#p]
        "boxes":  [b["boxes"] for b in batch],    # list of list[4]
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
    colors = get_colors()
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
    plt.show()