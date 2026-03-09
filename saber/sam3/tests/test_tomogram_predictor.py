"""
Test TomogramSAM3Adapter — run this on the GPU machine.

Tests (in order):
  1. Model loads without error (downloads from HuggingFace if needed)
  2. Inference state is created from a small synthetic tomogram
  3. A text prompt seeds segmentation on the middle Z-slice
  4. Bidirectional propagation runs and returns results for all Z-slices

Usage
-----
    # Download weights from HuggingFace (requires login or HF_TOKEN env var):
    python -m saber.sam3.tests.test_tomogram_predictor

    # Use a locally cached checkpoint:
    python -m saber.sam3.tests.test_tomogram_predictor --checkpoint /path/to/sam3.pt

    # Single test only:
    python -m saber.sam3.tests.test_tomogram_predictor --test load
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
    from saber.sam3.tomogram_predictor import TomogramSAM3Adapter

    t0 = time.time()
    adapter = TomogramSAM3Adapter(
        device="cuda",
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
    )
    elapsed = time.time() - t0

    print(f"  PASS  Model type    : {type(adapter.model).__name__}")
    print(f"  PASS  image_size    : {adapter.model.image_size}")
    print(f"  PASS  Load time     : {elapsed:.1f}s")
    print(f"  PASS  CUDA memory   : {torch.cuda.memory_allocated() // 1024**2} MiB allocated")
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
    state = adapter.create_inference_state_from_tomogram(tomogram)
    elapsed = time.time() - t0

    assert state["num_frames"] == 20, (
        f"Expected num_frames=20, got {state['num_frames']}"
    )
    assert "input_batch" in state, "Missing 'input_batch' key in inference_state"
    assert "feature_cache" in state, "Missing 'feature_cache' key in inference_state"
    assert not state["is_image_only"], "is_image_only should be False for tomograms"

    print(f"  PASS  num_frames    : {state['num_frames']}")
    print(f"  PASS  orig_height   : {state['orig_height']}")
    print(f"  PASS  orig_width    : {state['orig_width']}")
    print(f"  PASS  image_size    : {state['image_size']}")
    print(f"  PASS  Build time    : {elapsed:.2f}s")
    return state


def test_text_prompt(adapter, state):
    print()
    print("=" * 60)
    print("Test 3: Text prompt on seed Z-slice")
    print("=" * 60)

    seed_slice = state["num_frames"] // 2
    print(f"  Prompting slice {seed_slice} with text='organelle' ...")

    t0 = time.time()
    frame_idx, outputs = adapter.add_text_prompt(
        state, frame_idx=seed_slice, text="organelle"
    )
    elapsed = time.time() - t0

    print(f"  PASS  Returned frame_idx : {frame_idx}")
    print(f"  PASS  Output type        : {type(outputs).__name__}")
    print(f"  PASS  Prompt time        : {elapsed:.2f}s")

    if isinstance(outputs, dict):
        print(f"  INFO  Output keys        : {list(outputs.keys())}")

    return frame_idx


def test_propagation(adapter, state, seed_frame):
    print()
    print("=" * 60)
    print("Test 4: Bidirectional propagation through Z")
    print("=" * 60)

    print(f"  Propagating from slice {seed_frame} ...")
    t0 = time.time()
    results = adapter.propagate_bidirectional(state, start_frame_idx=seed_frame)
    elapsed = time.time() - t0

    assert len(results) > 0, "No results returned from propagation"

    frames_covered = sorted(results.keys())
    print(f"  PASS  Frames propagated  : {len(results)} / {state['num_frames']}")
    print(f"  PASS  Frame range        : {frames_covered[0]} – {frames_covered[-1]}")
    print(f"  PASS  Propagation time   : {elapsed:.2f}s")

    sample_key = frames_covered[0]
    sample_out = results[sample_key]
    print(f"  INFO  Sample output type : {type(sample_out).__name__}")
    if isinstance(sample_out, dict):
        print(f"  INFO  Sample output keys : {list(sample_out.keys())}")

    return results


def test_reset_state(adapter, state):
    print()
    print("=" * 60)
    print("Test 5: Reset state clears prompts")
    print("=" * 60)

    adapter.reset_state(state)
    # After reset, action_history should be empty
    assert len(state["action_history"]) == 0, "action_history not cleared by reset"
    assert len(state["tracker_inference_states"]) == 0, (
        "tracker_inference_states not cleared by reset"
    )
    print("  PASS  action_history cleared")
    print("  PASS  tracker_inference_states cleared")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

TESTS = {
    "load": test_model_loads,
    "state": test_inference_state,
    "prompt": test_text_prompt,
    "propagate": test_propagation,
    "reset": test_reset_state,
}


def main():
    parser = argparse.ArgumentParser(
        description="Test TomogramSAM3Adapter on a GPU machine."
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
        # Still need adapter for downstream tests
        from saber.sam3.tomogram_predictor import TomogramSAM3Adapter
        adapter = TomogramSAM3Adapter(
            device="cuda",
            checkpoint_path=args.checkpoint,
            load_from_HF=not args.no_hf,
        )

    if args.test == "state" or args.test is None:
        state = test_inference_state(adapter)
    else:
        rng = np.random.default_rng(42)
        state = adapter.create_inference_state_from_tomogram(
            rng.uniform(-1.0, 1.0, (20, 128, 128)).astype(np.float32)
        )

    if args.test == "prompt" or args.test is None:
        seed_frame = test_text_prompt(adapter, state)
    else:
        seed_frame = state["num_frames"] // 2

    if args.test == "propagate" or args.test is None:
        test_propagation(adapter, state, seed_frame)

    if args.test == "reset" or args.test is None:
        # Re-prompt first so reset has something to clear
        if args.test == "reset":
            adapter.add_text_prompt(state, frame_idx=seed_frame, text="organelle")
        test_reset_state(adapter, state)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
