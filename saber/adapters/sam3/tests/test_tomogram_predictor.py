"""
Test SAM3Adapter — run this on the GPU machine.

Tests (in order):
  1. Model loads without error (downloads from HuggingFace if needed)
  2. Inference state is created from a small synthetic tomogram
  3. A binary mask seeds segmentation on the middle Z-slice
  4. Bidirectional propagation runs and returns a (Z, H, W) uint16 volume
  5. Reset state clears prompts and tracking history

Usage
-----
    # Download weights from HuggingFace (requires login or HF_TOKEN env var):
    python -m saber.adapters.sam3.tests.test_tomogram_predictor

    # Use a locally cached checkpoint:
    python -m saber.adapters.sam3.tests.test_tomogram_predictor --checkpoint /path/to/sam3.pt

    # Single test only:
    python -m saber.adapters.sam3.tests.test_tomogram_predictor --test load
"""

import argparse
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def test_model_loads(checkpoint_path=None, load_from_HF=True):
    print("=" * 60)
    print("Test 1: Model loads without error")
    print("=" * 60)
    from saber.adapters.sam3 import SAM3Adapter
    from saber.adapters.base import SAM3AdapterConfig

    t0 = time.time()
    adapter = SAM3Adapter(
        config=SAM3AdapterConfig(
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
        ),
        device="cuda",
    )
    elapsed = time.time() - t0

    print(f"  PASS  Predictor type : {type(adapter.predictor).__name__}")
    print(f"  PASS  image_size     : {adapter.predictor.image_size}")
    print(f"  PASS  Load time      : {elapsed:.1f}s")
    print(f"  PASS  CUDA memory    : {torch.cuda.memory_allocated() // 1024**2} MiB allocated")
    return adapter


def test_inference_state(adapter):
    print()
    print("=" * 60)
    print("Test 2: Inference state from synthetic tomogram")
    print("=" * 60)

    # Small synthetic tomogram: 20 slices, 128×128 pixels
    rng = np.random.default_rng(42)
    tomogram = rng.uniform(-1.0, 1.0, (20, 128, 128)).astype(np.float32)

    t0 = time.time()
    adapter.set_volume(tomogram)
    elapsed = time.time() - t0
    state = adapter.inference_state

    assert state["num_frames"] == 20, (
        f"Expected num_frames=20, got {state['num_frames']}"
    )
    assert "images" in state, "Missing 'images' key in inference_state"
    assert state["images"].shape[0] == 20, (
        f"Expected 20 image frames, got {state['images'].shape[0]}"
    )

    print(f"  PASS  num_frames     : {state['num_frames']}")
    print(f"  PASS  video_height   : {state['video_height']}")
    print(f"  PASS  video_width    : {state['video_width']}")
    print(f"  PASS  images shape   : {list(state['images'].shape)}")
    print(f"  PASS  Build time     : {elapsed:.2f}s")
    return adapter


def test_mask_prompt(adapter):
    print()
    print("=" * 60)
    print("Test 3: Binary mask prompt on seed Z-slice")
    print("=" * 60)

    state = adapter.inference_state
    seed_slice = state["num_frames"] // 2
    H, W = state["video_height"], state["video_width"]

    # Synthetic circular mask in the center of the slice
    y, x = np.ogrid[:H, :W]
    mask = ((y - H // 2) ** 2 + (x - W // 2) ** 2) < (min(H, W) // 6) ** 2
    mask = mask.astype(np.float32)

    print(f"  Seeding slice {seed_slice} with a circular mask "
          f"({mask.sum():.0f} px foreground) ...")

    t0 = time.time()
    out = adapter.add_new_mask(
        frame_idx=seed_slice,
        obj_id=1,
        mask=mask,
    )
    elapsed = time.time() - t0

    frame_idx, obj_ids, low_res_masks, video_res_masks = out
    print(f"  PASS  Returned frame_idx : {frame_idx}")
    print(f"  PASS  obj_ids           : {obj_ids}")
    print(f"  PASS  video_res_masks shape : {list(video_res_masks.shape)}")
    print(f"  PASS  Prompt time       : {elapsed:.2f}s")

    return seed_slice


def test_propagation(adapter, seed_frame):
    print()
    print("=" * 60)
    print("Test 4: segment_volume (bidirectional propagation)")
    print("=" * 60)

    state = adapter.inference_state
    print(f"  Propagating from slice {seed_frame} ...")
    t0 = time.time()
    vol_masks = adapter.segment_volume(start_frame_idx=seed_frame)
    elapsed = time.time() - t0

    Z, H, W = adapter._vol_shape
    assert vol_masks.shape == (Z, H, W), (
        f"Expected shape {(Z, H, W)}, got {vol_masks.shape}"
    )
    assert vol_masks.dtype == np.uint16, f"Expected uint16, got {vol_masks.dtype}"

    nonzero_slices = int((vol_masks > 0).any(axis=(1, 2)).sum())
    print(f"  PASS  Output shape      : {list(vol_masks.shape)}")
    print(f"  PASS  Output dtype      : {vol_masks.dtype}")
    print(f"  PASS  Non-zero slices   : {nonzero_slices} / {Z}")
    print(f"  PASS  Propagation time  : {elapsed:.2f}s")
    return vol_masks


def test_reset_state(adapter):
    print()
    print("=" * 60)
    print("Test 5: Reset state clears prompts")
    print("=" * 60)

    # Re-prompt so reset has something to clear
    state = adapter.inference_state
    H, W = state["video_height"], state["video_width"]
    mask = np.ones((H, W), dtype=np.float32) * 0.5
    adapter.add_new_mask(frame_idx=0, obj_id=99, mask=mask)

    adapter.reset_state()

    state = adapter.inference_state
    assert len(state["obj_ids"]) == 0 or not state["tracking_has_started"], (
        "tracking_has_started should be False after reset"
    )
    print("  PASS  tracking_has_started cleared after reset")
    print("  PASS  State reset successfully")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

TESTS = {
    "load": test_model_loads,
    "state": test_inference_state,
    "prompt": test_mask_prompt,
    "propagate": test_propagation,
    "reset": test_reset_state,
}


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3Adapter on a GPU machine."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a local SAM3 checkpoint (.pt). "
             "If omitted, weights are downloaded from HuggingFace.",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable HuggingFace download (requires --checkpoint).",
    )
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()),
        default=None,
        help="Run only this test (default: run all).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Run on a GPU machine.")
        sys.exit(1)

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print()

    if args.test == "load" or args.test is None:
        adapter = test_model_loads(
            checkpoint_path=args.checkpoint,
            load_from_HF=not args.no_hf,
        )
    else:
        from saber.adapters.sam3 import SAM3Adapter
        from saber.adapters.base import SAM3AdapterConfig
        adapter = SAM3Adapter(
            config=SAM3AdapterConfig(
                checkpoint_path=args.checkpoint,
                load_from_HF=not args.no_hf,
            ),
            device="cuda",
        )

    rng = np.random.default_rng(42)
    tomogram = rng.uniform(-1.0, 1.0, (20, 128, 128)).astype(np.float32)

    if args.test == "state" or args.test is None:
        test_inference_state(adapter)
    else:
        adapter.set_volume(tomogram)

    if args.test == "prompt" or args.test is None:
        seed_frame = test_mask_prompt(adapter)
    else:
        seed_frame = adapter.inference_state["num_frames"] // 2

    if args.test == "propagate" or args.test is None:
        # Need a mask prompt before propagating
        if args.test == "propagate":
            state = adapter.inference_state
            H = state["video_height"]
            W = state["video_width"]
            y, x = np.ogrid[:H, :W]
            mask = ((y - H//2)**2 + (x - W//2)**2 < (min(H,W)//6)**2).astype(np.float32)
            adapter.add_new_mask(frame_idx=seed_frame, obj_id=1, mask=mask)
        test_propagation(adapter, seed_frame)

    if args.test == "reset" or args.test is None:
        test_reset_state(adapter)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
