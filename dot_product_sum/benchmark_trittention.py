#!/usr/bin/env python3
"""
Benchmarking script for trittention kernel performance.
Measures time and memory usage for Triton implementation.
Optionally benchmarks PyTorch implementation (may fail with large seq_len).
"""

import torch
import time
import gc
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import traceback

from trittention_triton import TrittentionTriton
from trittention_pytorch import Trittention_pytorch


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_forward_pass(
    model, q, k1, k2, v1, v2, num_warmup: int = 5, num_runs: int = 20
) -> Tuple[float, float]:
    """
    Benchmark forward pass performance.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(num_warmup):
        _ = model(q, k1, k2, v1, v2)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        output = model(q, k1, k2, v1, v2)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def benchmark_backward_pass(
    model, q, k1, k2, v1, v2, num_warmup: int = 5, num_runs: int = 20
) -> Tuple[float, float]:
    """
    Benchmark backward pass performance.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(num_warmup):
        output = model(q, k1, k2, v1, v2)
        if isinstance(output, tuple):
            output = output[0]  # PyTorch returns tuple
        grad_output = torch.randn_like(output)
        output.backward(grad_output, retain_graph=True)

        # Zero gradients
        for tensor in [q, k1, k2, v1, v2]:
            if tensor.grad is not None:
                tensor.grad.zero_()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        output = model(q, k1, k2, v1, v2)
        if isinstance(output, tuple):
            output = output[0]  # PyTorch returns tuple
        grad_output = torch.randn_like(output)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        output.backward(grad_output, retain_graph=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to ms

        # Zero gradients
        for tensor in [q, k1, k2, v1, v2]:
            if tensor.grad is not None:
                tensor.grad.zero_()

    return np.mean(times), np.std(times)


def run_benchmark(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    softmax_scale: float = 1 / 8,
    k_diff: int = 1024,
    benchmark_pytorch: bool = True,
    num_warmup: int = 5,
    num_runs: int = 20,
    input_precision: Optional[str] = None,
    convert_to_float32: bool = False,
) -> Dict[str, float]:
    """Run comprehensive benchmark for given configuration."""

    print(f"Benchmarking configuration:")
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Causal: {causal}, Device: {device}, Dtype: {dtype}")
    print(f"  Benchmark PyTorch: {benchmark_pytorch}")
    print()

    # Create models
    triton_model = TrittentionTriton(
        causal=causal,
        softmax_scale=softmax_scale,
        k_diff=k_diff,
        convert_to_float32=convert_to_float32,
        input_precision=input_precision,
    ).to(device=device, dtype=dtype)

    pytorch_model = None
    if benchmark_pytorch:
        pytorch_model = Trittention_pytorch(
            causal=causal, k_diff=k_diff, softmax_scale=softmax_scale
        ).to(device=device, dtype=dtype)

    # Generate input tensors
    torch.manual_seed(42)
    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k1 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k2 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v1 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v2 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    results = {}

    # Measure memory before
    memory_before = get_memory_usage()

    # Benchmark Triton implementation
    print("Benchmarking Triton implementation...")
    try:
        fwd_time_mean, fwd_time_std = benchmark_forward_pass(
            triton_model, q, k1, k2, v1, v2, num_warmup, num_runs
        )
        bwd_time_mean, bwd_time_std = benchmark_backward_pass(
            triton_model, q, k1, k2, v1, v2, num_warmup, num_runs
        )

        results["triton_forward_time_mean"] = fwd_time_mean
        results["triton_forward_time_std"] = fwd_time_std
        results["triton_backward_time_mean"] = bwd_time_mean
        results["triton_backward_time_std"] = bwd_time_std
        results["triton_total_time_mean"] = fwd_time_mean + bwd_time_mean

        print(f"  Forward:  {fwd_time_mean:.3f} ± {fwd_time_std:.3f} ms")
        print(f"  Backward: {bwd_time_mean:.3f} ± {bwd_time_std:.3f} ms")
        print(f"  Total:    {fwd_time_mean + bwd_time_mean:.3f} ms")

    except Exception as e:
        print(f"  Error: {str(e)}")
        results["triton_error"] = str(e)

    # Measure memory after Triton
    memory_after_triton = get_memory_usage()
    results["triton_memory_usage"] = memory_after_triton - memory_before

    # Benchmark PyTorch implementation if enabled
    if benchmark_pytorch and pytorch_model is not None:
        print("\nBenchmarking PyTorch implementation...")
        try:
            # Clone tensors for PyTorch to avoid interference
            q_pt = q.clone().detach().requires_grad_(True)
            k1_pt = k1.clone().detach().requires_grad_(True)
            k2_pt = k2.clone().detach().requires_grad_(True)
            v1_pt = v1.clone().detach().requires_grad_(True)
            v2_pt = v2.clone().detach().requires_grad_(True)

            fwd_time_mean, fwd_time_std = benchmark_forward_pass(
                pytorch_model, q_pt, k1_pt, k2_pt, v1_pt, v2_pt, num_warmup, num_runs
            )
            bwd_time_mean, bwd_time_std = benchmark_backward_pass(
                pytorch_model, q_pt, k1_pt, k2_pt, v1_pt, v2_pt, num_warmup, num_runs
            )

            results["pytorch_forward_time_mean"] = fwd_time_mean
            results["pytorch_forward_time_std"] = fwd_time_std
            results["pytorch_backward_time_mean"] = bwd_time_mean
            results["pytorch_backward_time_std"] = bwd_time_std
            results["pytorch_total_time_mean"] = fwd_time_mean + bwd_time_mean

            print(f"  Forward:  {fwd_time_mean:.3f} ± {fwd_time_std:.3f} ms")
            print(f"  Backward: {bwd_time_mean:.3f} ± {bwd_time_std:.3f} ms")
            print(f"  Total:    {fwd_time_mean + bwd_time_mean:.3f} ms")

            # Compute speedup if both implementations succeeded
            if "triton_total_time_mean" in results:
                speedup = (
                    results["pytorch_total_time_mean"]
                    / results["triton_total_time_mean"]
                )
                results["triton_speedup"] = speedup
                print(f"\nTriton speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"  Error: {str(e)}")
            print(f"  Error traceback: {traceback.format_exc()}")
            results["pytorch_error"] = str(e)

    # Measure memory after PyTorch
    memory_after_pytorch = get_memory_usage()
    if benchmark_pytorch:
        results["pytorch_memory_usage"] = memory_after_pytorch - memory_after_triton

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="Benchmark trittention kernel performance"
    )
    parser.add_argument(
        "--no-pytorch", action="store_true", help="Skip PyTorch benchmarking"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "float32"], help="Data type"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup runs"
    )
    parser.add_argument(
        "--num-runs", type=int, default=20, help="Number of benchmark runs"
    )

    args = parser.parse_args()

    benchmark_pytorch = not args.no_pytorch
    device = args.device

    print("Trittention Kernel Benchmark")
    print("=" * 50)

    input_precision = None  # "tf32x3"
    convert_to_float32 = False

    # Test configurations
    configs = [
        # Small configurations
        {
            "batch_size": 2,
            "seq_len": 64,
            "num_heads": 4,
            "head_dim": 32,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 1024,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 2,
            "seq_len": 128,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 1024,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        # Medium configurations
        {
            "batch_size": 4,
            "seq_len": 256,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 1024,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 4,
            "seq_len": 256,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 1024,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 4,
            "seq_len": 256,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 40,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 4,
            "seq_len": 256,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 40,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 2,
            "seq_len": 512,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        # Large configurations (PyTorch may fail here)
        {
            "batch_size": 1,
            "seq_len": 1024,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 2048,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 1024,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 2048,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 128,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 2048,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 128,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 2048,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.float32,
            "k_diff": 128,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 2048,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 128,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 2048,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
        {
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 8,
            "head_dim": 64,
            "causal": True,
            "dtype": torch.bfloat16,
            "k_diff": 128,
            "input_precision": input_precision,
            "convert_to_float32": convert_to_float32,
        },
    ]

    # Run benchmarks
    all_results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*20} Configuration {i+1} {'='*20}")

        # For large seq_len, disable PyTorch benchmarking to avoid OOM
        config_benchmark_pytorch = benchmark_pytorch and config["seq_len"] <= 512

        if config["seq_len"] > 512 and benchmark_pytorch:
            print("Large seq_len detected, disabling PyTorch benchmarking to avoid OOM")

        try:
            results = run_benchmark(
                device=device,
                benchmark_pytorch=config_benchmark_pytorch,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                **config,
            )
            results.update(config)
            all_results.append(results)

        except Exception as e:
            print(f"Error in configuration {i+1}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    print(
        f"{'Config':<8} {'Seq Len':<8} {'K_diff':<8} {'Dtype':<12} {'Triton (ms)':<12} {'PyTorch (ms)':<13} {'Speedup':<8}"
    )
    print("-" * 80)

    for i, results in enumerate(all_results):
        seq_len = results.get("seq_len", "N/A")
        k_diff = results.get("k_diff", "N/A")
        dtype = results.get("dtype", "N/A")
        triton_time = results.get("triton_total_time_mean", None)
        pytorch_time = results.get("pytorch_total_time_mean", None)
        speedup = results.get("triton_speedup", None)

        triton_str = f"{triton_time:.1f}" if triton_time is not None else "FAIL"
        pytorch_str = f"{pytorch_time:.1f}" if pytorch_time is not None else "SKIP/FAIL"
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        dtype_str = str(dtype).replace("torch.", "") if dtype != "N/A" else "N/A"

        print(
            f"{i+1:<8} {seq_len:<8} {k_diff:<8} {dtype_str:<12} {triton_str:<12} {pytorch_str:<13} {speedup_str:<8}"
        )

    print("\n" + "=" * 50)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
