#!/usr/bin/env python3
"""
Simplified comparison script for Flash Trittention vs PyTorch reference.

This script mirrors the structure of src/flash_trittention/triple_dot_product/test_trittention.py:
- All configuration variables are declared together at the top of the file
- Forward and backward comparisons are done in a single flow
"""

import os
import sys
import torch
import torch.nn.functional as F

# Import from triple_dot_product (comment out dot_product_sum when using this)
from flash_trittention.triple_dot_product.flash_trittention import flash_trittention
from flash_trittention.triple_dot_product.trittention_pytorch import (
    Trittention_pytorch,
)


# ==========================
# Global configuration block
# ==========================
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # or torch.bfloat16

BATCH_SIZE = 2
NUM_HEADS = 2
SEQ_LEN = 202
HEAD_DIM = 64
SCALE_INPUT = 1.0

SOFTMAX_SCALE = 1.0 / (HEAD_DIM**0.5)
CAUSAL = True
K1_WINDOW = 100
K2_WINDOW = 100
KK_LEFT = 32
KK_RIGHT = 2


def print_stats(tensor: torch.Tensor, name: str = "") -> None:
    print(f"\n{name} stats:")
    print(f"Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    print(
        f"Abs mean: {tensor.abs().mean().item():.6f}, Abs Std: {tensor.abs().std().item():.6f}"
    )
    print(f"Max: {tensor.max().item():.6f}, Min: {tensor.min().item():.6f}")
    print(
        f"Number of elements < 1e-4: {(tensor.abs() < 1e-4).sum()}, "
        f"percentage: {(tensor.abs() < 1e-4).sum() / tensor.numel() * 100:.2f}%"
    )
    print(
        f"Number of elements < 1e-2: {(tensor.abs() < 1e-2).sum()}, "
        f"percentage: {(tensor.abs() < 1e-2).sum() / tensor.numel() * 100:.2f}%"
    )


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name1: str,
    name2: str,
) -> bool:
    print(f"\n{'='*60}")
    print(f"{name1.upper()} vs {name2.upper()} COMPARISON")
    print(f"{'='*60}")

    print_stats(tensor1, name1)
    print_stats(tensor2, name2)

    diff = tensor1 - tensor2
    print(f"\nMax absolute difference: {diff.abs().max().item():.6f}")
    print(f"Mean absolute difference: {diff.abs().mean().item():.6f}")
    print_stats(diff, f"{name1} - {name2}")

    reference_scale = tensor1.abs().mean() * 0.2
    mean_ratio = ((diff.abs()) / (reference_scale + tensor1.abs())).mean().item()
    print(f"Mean ratio: {mean_ratio:.6f}")

    # 1% error is ok (imo)
    return mean_ratio < 1e-2


def main() -> None:
    print("Flash Trittention vs PyTorch reference")
    print("=" * 80)
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, running on CPU (this will be slow)")

    # Create base inputs
    Q = SCALE_INPUT * torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    K1 = SCALE_INPUT * torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    K2 = SCALE_INPUT * torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    V1 = SCALE_INPUT * torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    V2 = SCALE_INPUT * torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    print("Running with configuration:")
    print(
        f"B={BATCH_SIZE}, H={NUM_HEADS}, T={SEQ_LEN}, D={HEAD_DIM}, device={DEVICE}, dtype={DTYPE}"
    )

    # =============================
    # PyTorch reference fwd + bwd
    # =============================
    print(f"\n{'='*80}")
    print("PYTORCH REFERENCE FORWARD AND BACKWARD PASS")
    print(f"{'='*80}")

    model_pt = Trittention_pytorch(
        causal=CAUSAL,
        softmax_scale=SOFTMAX_SCALE,
        n_ctx=SEQ_LEN,
        k1_window=K1_WINDOW,
        k2_window=K2_WINDOW,
        kk_left=KK_LEFT,
        kk_right=KK_RIGHT,
    )

    Q_pt = Q.clone().detach().requires_grad_(True)
    K1_pt = K1.clone().detach().requires_grad_(True)
    K2_pt = K2.clone().detach().requires_grad_(True)
    V1_pt = V1.clone().detach().requires_grad_(True)
    V2_pt = V2.clone().detach().requires_grad_(True)

    out_pt, _, _, M_pt, L_pt = model_pt(Q_pt, K1_pt, K2_pt, V1_pt, V2_pt)
    dO = torch.randn_like(out_pt)
    out_pt.backward(dO)
    dq_pt, dk1_pt, dk2_pt, dv1_pt, dv2_pt = (
        Q_pt.grad,
        K1_pt.grad,
        K2_pt.grad,
        V1_pt.grad,
        V2_pt.grad,
    )

    # =============================
    # Flash trittention fwd + bwd
    # =============================
    print(f"\n{'='*80}")
    print("FLASH TRITTENTION FORWARD AND BACKWARD PASS")
    print(f"{'='*80}")

    Q_fl = Q.clone().detach().requires_grad_(True)
    K1_fl = K1.clone().detach().requires_grad_(True)
    K2_fl = K2.clone().detach().requires_grad_(True)
    V1_fl = V1.clone().detach().requires_grad_(True)
    V2_fl = V2.clone().detach().requires_grad_(True)

    out_fl = flash_trittention(
        Q_fl,
        K1_fl,
        K2_fl,
        V1_fl,
        V2_fl,
        softmax_scale=SOFTMAX_SCALE,
        kk_left=KK_LEFT,
        kk_right=KK_RIGHT,
        k1_window=K1_WINDOW,
        k2_window=K2_WINDOW,
        causal=CAUSAL,
        convert_to_float32=True,
        input_precision=None,
    )
    out_fl.backward(dO)
    dq_fl, dk1_fl, dk2_fl, dv1_fl, dv2_fl = (
        Q_fl.grad,
        K1_fl.grad,
        K2_fl.grad,
        V1_fl.grad,
        V2_fl.grad,
    )

    # =============================
    # Comparisons and summary
    # =============================
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    results = {
        "output": compare_tensors(out_pt, out_fl, "PyTorch Output", "Flash Output"),
        "dq": compare_tensors(dq_pt, dq_fl, "PyTorch dQ", "Flash dQ"),
        "dk1": compare_tensors(dk1_pt, dk1_fl, "PyTorch dK1", "Flash dK1"),
        "dk2": compare_tensors(dk2_pt, dk2_fl, "PyTorch dK2", "Flash dK2"),
        "dv1": compare_tensors(dv1_pt, dv1_fl, "PyTorch dV1", "Flash dV1"),
        "dv2": compare_tensors(dv2_pt, dv2_fl, "PyTorch dV2", "Flash dV2"),
    }

    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name.upper():>10}: {status}")
        if not passed:
            all_passed = False

    print(f"\n{'='*80}")
    print(
        "🎉 ALL TESTS PASSED! Flash Trittention matches PyTorch reference."
        if all_passed
        else "❌ SOME TESTS FAILED! Check the differences above."
    )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
