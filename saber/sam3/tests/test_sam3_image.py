"""
SAM3 2D image segmentation test — run on the GPU machine.

This script validates the full SAM3 text-prompted segmentation pipeline
on a 2D image before committing to the 3D tomogram integration.

It diagnoses the most common failure in prior prototypes:
  - Passing a grayscale (H, W) numpy array to Sam3Processor.set_image()
    produces a 1-channel tensor; SAM3's backbone expects 3 channels.
  Fix: convert to (H, W, 3) before calling set_image().

Usage
-----
    # Minimal test with a synthetic image (no file I/O):
    python -m saber.sam3.tests.test_sam3_image

    # Test on a real image file (TIFF, PNG, JPG):
    python -m saber.sam3.tests.test_sam3_image --image /path/to/image.tif

    # Pass a local BPE vocab (if SAM3 pkg_resources lookup fails):
    python -m saber.sam3.tests.test_sam3_image --bpe /path/to/bpe_simple_vocab_16e6.txt.gz

    # Save the visualisation to disk (no display needed on HPC):
    python -m saber.sam3.tests.test_sam3_image --image /path/to/image.tif --save output.png
"""

import argparse
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_rgb_float32(image: np.ndarray) -> np.ndarray:
    """
    Convert any 2D grayscale or 3D array to an (H, W, 3) float32 array
    in [0, 1].  This is the correct input format for Sam3Processor.set_image.

    Sam3Processor uses torchvision v2.functional.to_image() internally,
    which expects:
      - (H, W, 3) uint8  — RGB image
      - (H, W, 3) float  — RGB image (values in [0, 1])
      - (H, W, C) tensor — any channel count handled by to_image
    A bare (H, W) array creates a 1-channel tensor that breaks SAM3's backbone.
    """
    if image.ndim == 2:
        # Grayscale → RGB by repeating the channel
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            f"Expected (H,W), (H,W,1) or (H,W,3), got shape {image.shape}"
        )

    # Ensure float32 in [0, 1]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Apply the same preprocessing used in cryoTomoSegmenter.generate_slab:
      1. Local contrast normalisation (removes background variation)
      2. Min-max normalisation to [0, 1]
      3. Grayscale → RGB
    """
    from saber.utils.preprocessing import contrast, normalize

    image = contrast(image, std_cutoff=3)
    image = normalize(image, rgb=False)
    return _to_rgb_float32(image)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_loads(bpe_path=None):
    print("=" * 60)
    print("Test 1: SAM3 image model and processor load")
    print("=" * 60)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, device="cuda")
    elapsed = time.time() - t0

    print(f"  PASS  Model type   : {type(model).__name__}")
    print(f"  PASS  image_size   : {processor.resolution}")
    print(f"  PASS  Load time    : {elapsed:.1f}s")
    print(f"  PASS  CUDA memory  : {torch.cuda.memory_allocated() // 1024**2} MiB")
    return model, processor


def test_set_image(processor, image_rgb: np.ndarray):
    """
    Verify that set_image accepts an (H, W, 3) float32 array without error
    and populates backbone features.
    """
    print()
    print("=" * 60)
    print("Test 2: set_image with (H, W, 3) float32 input")
    print("=" * 60)

    print(f"  Input shape : {image_rgb.shape}  dtype: {image_rgb.dtype}")
    print(f"  Value range : [{image_rgb.min():.3f}, {image_rgb.max():.3f}]")

    t0 = time.time()
    state = processor.set_image(image_rgb)
    elapsed = time.time() - t0

    assert "backbone_out" in state, "set_image did not populate backbone_out"
    sam2_out = state["backbone_out"].get("sam2_backbone_out", {})
    features = sam2_out.get("vision_features")
    assert features is not None, (
        "sam2_backbone_out['vision_features'] not found — "
        "check enable_inst_interactivity flag"
    )

    print(f"  PASS  backbone_out populated")
    print(f"  PASS  vision_features shape : {list(features.shape)}")
    print(f"  PASS  set_image time        : {elapsed:.2f}s")
    return state


def test_text_prompt(processor, state, prompt="cell"):
    print()
    print("=" * 60)
    print(f"Test 3: Text prompt '{prompt}'")
    print("=" * 60)

    t0 = time.time()
    output = processor.set_text_prompt(state=state, prompt=prompt)
    elapsed = time.time() - t0

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    assert masks is not None, "No 'masks' key in output"
    print(f"  PASS  Detections found : {len(masks)}")
    print(f"  PASS  Prompt time      : {elapsed:.2f}s")

    if len(masks) > 0:
        print(f"  INFO  Mask shape       : {list(masks.shape)}")
        print(f"  INFO  Score range      : [{scores.min().item():.3f}, {scores.max().item():.3f}]")
    else:
        print("  INFO  No objects detected above confidence threshold.")
        print("        Try a different prompt or lower confidence_threshold.")

    return masks, boxes, scores


def test_grayscale_fail_then_fix(processor):
    """
    Demonstrate the original failure mode: passing a bare (H, W) grayscale
    array crashes or silently produces wrong results.
    Then show the fix.
    """
    print()
    print("=" * 60)
    print("Test 4: Grayscale (H,W) input — expected failure + fix")
    print("=" * 60)

    rng = np.random.default_rng(1)
    gray = rng.uniform(0, 1, (256, 256)).astype(np.float32)

    print(f"  Attempting set_image with bare (H, W) array ...")
    try:
        state_bad = processor.set_image(gray)
        processor.set_text_prompt(state=state_bad, prompt="cell")
        print("  WARN  No exception raised — check if backbone accepted 1-channel input")
    except Exception as e:
        print(f"  PASS  Correctly fails with : {type(e).__name__}: {str(e)[:80]}")

    print(f"  Retrying with (H, W, 3) conversion ...")
    gray_rgb = _to_rgb_float32(gray)
    state_ok = processor.set_image(gray_rgb)
    output = processor.set_text_prompt(state=state_ok, prompt="object")
    print(f"  PASS  (H, W, 3) input works — {len(output['masks'])} detections")


def test_save_visualisation(image_rgb, masks, scores, save_path=None):
    print()
    print("=" * 60)
    print("Test 5: Visualisation")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for HPC
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP  matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input image (preprocessed)")
    axes[0].axis("off")

    if masks is not None and len(masks) > 0:
        masks_np = masks.detach().cpu().numpy()  # (N, 1, H, W)
        overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.float32)
        colors = plt.cm.tab10.colors
        for i, (m, s) in enumerate(zip(masks_np[:, 0], scores.cpu())):
            color = colors[i % len(colors)]
            overlay[m > 0.5] = [*color[:3], 0.45]
        axes[1].imshow(image_rgb)
        axes[1].imshow(overlay)
        axes[1].set_title(f"SAM3 detections ({len(masks)} objects)")
    else:
        axes[1].imshow(image_rgb)
        axes[1].set_title("No detections")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  PASS  Saved to {save_path}")
    else:
        print("  INFO  Pass --save /path/output.png to write visualisation to disk")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3 2D image segmentation on a GPU machine."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a TIFF/PNG/JPG image to test on. "
             "If omitted, a synthetic image is used.",
    )
    parser.add_argument(
        "--bpe",
        default=None,
        help="Path to bpe_simple_vocab_16e6.txt.gz. "
             "Leave blank if sam3 is pip-installed (pkg_resources resolves it).",
    )
    parser.add_argument(
        "--prompt",
        default="cell",
        help="Text prompt to pass to SAM3 (default: 'cell').",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save visualisation PNG to this path instead of displaying.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Run on a GPU machine.")
        sys.exit(1)

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print()

    # ---- Load model ----
    model, processor = test_model_loads(bpe_path=args.bpe)
    processor.confidence_threshold = args.confidence

    # ---- Load / generate image ----
    if args.image is not None:
        import skimage.io as sio
        raw = sio.imread(args.image)
        print(f"\n  Loaded image: {raw.shape}  dtype: {raw.dtype}")
        # Handle multi-channel TIFFs
        if raw.ndim == 3 and raw.shape[0] < raw.shape[-1]:
            raw = raw[0]  # take first channel if (C,H,W)
        image_rgb = _preprocess(raw)
    else:
        print("\n  No --image provided; using synthetic 512×512 image.")
        rng = np.random.default_rng(42)
        # Synthetic image with a few bright blobs (simulate cells/organelles)
        base = rng.uniform(0, 0.2, (512, 512)).astype(np.float32)
        for _ in range(8):
            cy, cx = rng.integers(80, 430, 2)
            r = rng.integers(20, 50)
            y, x = np.ogrid[:512, :512]
            base[((y - cy) ** 2 + (x - cx) ** 2) < r ** 2] += rng.uniform(0.4, 0.8)
        image_rgb = _to_rgb_float32(np.clip(base, 0, 1))

    # ---- Run tests ----
    state = test_set_image(processor, image_rgb)
    masks, boxes, scores = test_text_prompt(processor, state, prompt=args.prompt)
    test_grayscale_fail_then_fix(processor)
    test_save_visualisation(image_rgb, masks, scores, save_path=args.save)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
