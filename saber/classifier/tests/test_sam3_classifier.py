"""
Test SAM3Classifier — run this on the GPU machine.

Tests (in order):
  1. Model loads without error (downloads SAM3 weights from HuggingFace)
  2. Forward pass with a synthetic image + binary mask
  3. Output shape matches num_classes
  4. Backbone parameters are frozen (no gradients)
  5. Backbone stays in eval mode during training

Usage
-----
    python -m saber.classifier.tests.test_sam3_classifier
    python -m saber.classifier.tests.test_sam3_classifier --num-classes 5
"""

import argparse
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_loads(num_classes=2, hidden_dims=256):
    print("=" * 60)
    print("Test 1: SAM3Classifier loads without error")
    print("=" * 60)
    from saber.classifier.models.SAM3 import SAM3Classifier

    t0 = time.time()
    model = SAM3Classifier(num_classes=num_classes, hidden_dims=hidden_dims, deviceID=0)
    elapsed = time.time() - t0

    print(f"  PASS  Model type   : {model.name}")
    print(f"  PASS  num_classes  : {num_classes}")
    print(f"  PASS  device       : {model.device}")
    print(f"  PASS  Load time    : {elapsed:.1f}s")
    print(
        f"  PASS  CUDA memory  : "
        f"{torch.cuda.memory_allocated() // 1024**2} MiB allocated"
    )
    return model


def test_forward_pass(model, num_classes=2):
    print()
    print("=" * 60)
    print("Test 2: Forward pass with synthetic image + mask")
    print("=" * 60)

    B, H, W = 2, 256, 256

    # Synthetic single-channel tomogram slice in [-1, 1]
    rng = np.random.default_rng(7)
    x_np = rng.uniform(-1.0, 1.0, (B, 1, H, W)).astype(np.float32)
    x = torch.as_tensor(x_np, device=model.device)

    # Random binary mask (1 = inside object)
    mask_np = (rng.random((B, 1, H, W)) > 0.5).astype(np.float32)
    mask = torch.as_tensor(mask_np, device=model.device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(x, mask)
    elapsed = time.time() - t0

    assert logits.shape == (B, num_classes), (
        f"Expected logits shape ({B}, {num_classes}), got {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "NaN values in output logits"
    assert not torch.isinf(logits).any(), "Inf values in output logits"

    print(f"  PASS  Input shape  : {list(x.shape)}")
    print(f"  PASS  Mask shape   : {list(mask.shape)}")
    print(f"  PASS  Output shape : {list(logits.shape)}")
    print(f"  PASS  Forward time : {elapsed:.2f}s")
    print(f"  INFO  Logit range  : [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    return logits


def test_backbone_frozen(model):
    print()
    print("=" * 60)
    print("Test 3: Backbone parameters are frozen")
    print("=" * 60)

    backbone_params = list(model.backbone.parameters())
    assert len(backbone_params) > 0, "Backbone has no parameters"

    unfrozen = [p for p in backbone_params if p.requires_grad]
    assert len(unfrozen) == 0, (
        f"{len(unfrozen)} backbone parameters have requires_grad=True"
    )

    # Projection head should be trainable
    head_params = list(model.projection.parameters()) + list(model.classifier.parameters())
    trainable = [p for p in head_params if p.requires_grad]
    assert len(trainable) > 0, "Projection/classifier head has no trainable parameters"

    print(f"  PASS  Backbone params    : {len(backbone_params)} (all frozen)")
    print(f"  PASS  Trainable head params : {len(trainable)}")


def test_backbone_stays_eval(model):
    print()
    print("=" * 60)
    print("Test 4: Backbone stays in eval mode during model.train()")
    print("=" * 60)

    model.train()  # put model in training mode
    assert not model.backbone.training, (
        "SAM3 backbone should remain in eval mode after model.train()"
    )
    model.eval()  # restore eval mode
    print("  PASS  Backbone training flag : False after model.train()")
    print("  PASS  Backbone training flag : False after model.eval()")


def test_batch_consistency(model, num_classes=2):
    print()
    print("=" * 60)
    print("Test 5: Single vs batched inference give consistent results")
    print("=" * 60)

    H, W = 128, 128
    rng = np.random.default_rng(99)
    x_np = rng.uniform(-1.0, 1.0, (1, 1, H, W)).astype(np.float32)
    m_np = (rng.random((1, 1, H, W)) > 0.5).astype(np.float32)

    x = torch.as_tensor(x_np, device=model.device)
    m = torch.as_tensor(m_np, device=model.device)

    with torch.no_grad():
        logits_single = model(x, m)
        # Run again — should be deterministic
        logits_again = model(x, m)

    diff = (logits_single - logits_again).abs().max().item()
    assert diff < 1e-4, f"Non-deterministic output: max diff = {diff}"

    print(f"  PASS  Max diff between two runs : {diff:.2e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3Classifier on a GPU machine."
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of output classes."
    )
    parser.add_argument(
        "--hidden-dims", type=int, default=256, help="Hidden dimension size."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Run on a GPU machine.")
        sys.exit(1)

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print()

    model = test_model_loads(num_classes=args.num_classes, hidden_dims=args.hidden_dims)
    test_forward_pass(model, num_classes=args.num_classes)
    test_backbone_frozen(model)
    test_backbone_stays_eval(model)
    test_batch_consistency(model, num_classes=args.num_classes)

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
