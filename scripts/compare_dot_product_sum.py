#!/usr/bin/env python3
"""
Comprehensive comparison script for trittention forward and backward passes.
Compares Triton implementation against PyTorch autograd reference.
Both implementations take 5 tensors: q, k1, k2, v1, v2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, Optional
import sys
import os

# Current directory is already dot_product_sum, no need to add path

from flash_trittention.dot_product_sum.trittention_triton import TrittentionTriton
from flash_trittention.dot_product_sum.trittention_pytorch import Trittention_pytorch


def compute_statistics(
    triton_val: torch.Tensor, pytorch_val: torch.Tensor, name: str
) -> Dict[str, float]:
    """Compute comparison statistics between triton and pytorch tensors."""

    # Ensure tensors are on same device and dtype
    triton_val = triton_val.float().cpu()
    pytorch_val = pytorch_val.float().cpu()

    # Absolute values
    triton_abs = torch.abs(triton_val)
    pytorch_abs = torch.abs(pytorch_val)

    # Absolute difference
    abs_diff = torch.abs(triton_val - pytorch_val)

    # Relative difference (avoid division by zero)
    eps = 1e-8
    rel_diff = abs_diff / (torch.abs(pytorch_val) + eps)

    stats = {
        f"{name}_triton_mean_abs": triton_abs.mean().item(),
        f"{name}_triton_max_abs": triton_abs.max().item(),
        f"{name}_pytorch_mean_abs": pytorch_abs.mean().item(),
        f"{name}_pytorch_max_abs": pytorch_abs.max().item(),
        f"{name}_abs_diff_mean": abs_diff.mean().item(),
        f"{name}_abs_diff_max": abs_diff.max().item(),
        f"{name}_rel_diff_mean": rel_diff.mean().item(),
        f"{name}_rel_diff_max": rel_diff.max().item(),
    }

    return stats


def run_comparison(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    causal: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    softmax_scale: float = 1 / 8,
    k_diff: int = 1024,
    input_precision: Optional[str] = None,
    convert_to_float32: bool = False,
    input_std: float = 1.0,
    grad_std: float = 1.0,
) -> Dict[str, Any]:
    """Run comprehensive comparison between Triton and PyTorch implementations."""

    print(f"Running comparison with:")
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Causal: {causal}, Device: {device}, Dtype: {dtype}")
    print()

    # Initialize models
    triton_model = TrittentionTriton(
        causal=causal,
        softmax_scale=softmax_scale,
        k_diff=k_diff,
        convert_to_float32=convert_to_float32,
        input_precision=input_precision,
    ).to(device=device, dtype=dtype)
    pytorch_model = Trittention_pytorch(
        causal=causal, k_diff=k_diff, softmax_scale=softmax_scale
    ).to(device=device, dtype=dtype)

    # Generate random input tensors: q, k1, k2, v1, v2
    torch.manual_seed(42)
    q = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        * input_std
    )
    k1 = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        * input_std
    )
    k2 = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        * input_std
    )
    v1 = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        * input_std
    )
    v2 = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        * input_std
    )

    q = q.clone().detach().requires_grad_(True)
    k1 = k1.clone().detach().requires_grad_(True)
    k2 = k2.clone().detach().requires_grad_(True)
    v1 = v1.clone().detach().requires_grad_(True)
    v2 = v2.clone().detach().requires_grad_(True)

    # Clone tensors for PyTorch model (to ensure same inputs but separate gradients)
    q_pytorch = q.clone().detach().requires_grad_(True)
    k1_pytorch = k1.clone().detach().requires_grad_(True)
    k2_pytorch = k2.clone().detach().requires_grad_(True)
    v1_pytorch = v1.clone().detach().requires_grad_(True)
    v2_pytorch = v2.clone().detach().requires_grad_(True)

    # Forward pass
    print("=== FORWARD PASS ===")

    # Triton forward
    triton_out, triton_m = triton_model(q, k1, k2, v1, v2)

    # PyTorch forward (returns multiple values: output, pre_softmax_attn_score, v_gated, M)
    pytorch_out, pytorch_pre_softmax, pytorch_v_gated, pytorch_m = pytorch_model(
        q_pytorch, k1_pytorch, k2_pytorch, v1_pytorch, v2_pytorch
    )

    # Forward pass statistics
    stats = {}

    # Output comparison
    output_stats = compute_statistics(triton_out, pytorch_out, "output")
    stats.update(output_stats)

    # M comparison between Triton and PyTorch
    m_stats = compute_statistics(triton_m, pytorch_m, "M")
    stats.update(m_stats)

    print("Forward pass statistics:")
    for key, value in output_stats.items():
        if "output" in key:
            print(f"  {key}: {value:.6e}")

    print("M (row-wise max) comparison:")
    for key, value in m_stats.items():
        if "M" in key:
            print(f"  {key}: {value:.6e}")

    # Backward pass
    print("\n=== BACKWARD PASS ===")

    # Create gradient for backward pass - use same dO for both
    grad_output = torch.randn_like(triton_out) * grad_std
    # Triton backward
    triton_out.backward(grad_output.clone(), retain_graph=True)
    triton_grad_q = q.grad.clone()
    triton_grad_k1 = k1.grad.clone()
    triton_grad_k2 = k2.grad.clone()
    triton_grad_v1 = v1.grad.clone()
    triton_grad_v2 = v2.grad.clone()

    # PyTorch backward
    pytorch_out.backward(grad_output.clone(), retain_graph=True)
    pytorch_grad_q = q_pytorch.grad.clone()
    pytorch_grad_k1 = k1_pytorch.grad.clone()
    pytorch_grad_k2 = k2_pytorch.grad.clone()
    pytorch_grad_v1 = v1_pytorch.grad.clone()
    pytorch_grad_v2 = v2_pytorch.grad.clone()

    # Gradient comparisons
    print("Backward pass statistics:")

    # dQ comparison
    dq_stats = compute_statistics(triton_grad_q, pytorch_grad_q, "dQ")
    stats.update(dq_stats)
    print("dQ (query gradient):")
    for key, value in dq_stats.items():
        if "dQ" in key:
            print(f"  {key}: {value:.6e}")

    # dK1 comparison
    dk1_stats = compute_statistics(triton_grad_k1, pytorch_grad_k1, "dK1")
    stats.update(dk1_stats)
    print("dK1 (key1 gradient):")
    for key, value in dk1_stats.items():
        if "dK1" in key:
            print(f"  {key}: {value:.6e}")

    # dK2 comparison
    dk2_stats = compute_statistics(triton_grad_k2, pytorch_grad_k2, "dK2")
    stats.update(dk2_stats)
    print("dK2 (key2 gradient):")
    for key, value in dk2_stats.items():
        if "dK2" in key:
            print(f"  {key}: {value:.6e}")

    # dV1 comparison
    dv1_stats = compute_statistics(triton_grad_v1, pytorch_grad_v1, "dV1")
    stats.update(dv1_stats)
    print("dV1 (value1 gradient):")
    for key, value in dv1_stats.items():
        if "dV1" in key:
            print(f"  {key}: {value:.6e}")

    # dV2 comparison
    dv2_stats = compute_statistics(triton_grad_v2, pytorch_grad_v2, "dV2")
    stats.update(dv2_stats)
    print("dV2 (value2 gradient):")
    for key, value in dv2_stats.items():
        if "dV2" in key:
            print(f"  {key}: {value:.6e}")

    return stats


def main():
    """Main comparison function."""
    print("Trittention Forward/Backward Pass Comparison")
    print("=" * 50)

    input_precision = "tf32x3"
    convert_to_float32 = True

    input_std = 4.0
    grad_std = 1.0

    # Test configurations
    configs = [
        {
            "batch_size": 2,
            "seq_len": 128,
            "num_heads": 4,
            "head_dim": 64,
            "causal": False,
            "softmax_scale": 1 / 8,
            "k_diff": 1024,
            "dtype": torch.bfloat16,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 128,
            "num_heads": 8,
            "head_dim": 64,
            "causal": False,
            "softmax_scale": 1 / 8,
            "k_diff": 1024,
            "dtype": torch.bfloat16,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 128,
            "num_heads": 8,
            "head_dim": 64,
            "causal": False,
            "softmax_scale": 1 / 8,
            "k_diff": 1024,
            "dtype": torch.float32,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 200,
            "num_heads": 8,
            "head_dim": 32,
            "causal": False,
            "softmax_scale": 1 / 16,
            "k_diff": 16,
            "dtype": torch.bfloat16,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 200,
            "num_heads": 8,
            "head_dim": 128,
            "causal": True,
            "softmax_scale": 1 / 16,
            "k_diff": 16,
            "dtype": torch.float32,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 350,
            "num_heads": 8,
            "head_dim": 32,
            "causal": False,
            "softmax_scale": 1 / 8,
            "k_diff": 50,
            "dtype": torch.float32,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    for i, config in enumerate(configs):
        print(f"\n{'='*20} Configuration {i+1} {'='*20}")
        try:
            stats = run_comparison(
                device=device,
                **config,
                input_std=input_std,
                grad_std=grad_std,
            )
            print(f"\nConfiguration {i+1} completed successfully!")

        except Exception as e:
            print(f"Error in configuration {i+1}: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("Comparison completed!")


if __name__ == "__main__":
    main()
