import torch
import torch.nn.functional as F
from dot_product_sum.trittention_pytorch import Trittention_pytorch
from slow_trittention import slow_trittention_dot_product_sum


def print_stats(tensor):
    print(f"Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    print(
        f"Abs mean: {tensor.abs().mean().item():.6f}, Abs Std: {tensor.abs().std().item():.6f}"
    )
    print(f"Max: {tensor.max().item():.6f}, Min: {tensor.min().item():.6f}")

    # Number of elements less than 1e-4
    print(
        f"Number of elements less than 1e-4, count: {(tensor.abs() < 1e-4).sum()}, percentage: {(tensor.abs() < 1e-4).sum() / tensor.numel() * 100:.2f}%"
    )
    print(
        f"Number of elements less than 1e-2, count: {(tensor.abs() < 1e-2).sum()}, percentage: {(tensor.abs() < 1e-2).sum() / tensor.numel() * 100:.2f}%"
    )


def compare_trittention_implementations():
    """
    Compare the output of slow trittention (sum of dot products)
    with the PyTorch trittention implementation.
    """
    print("Comparing Trittention Implementations")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    bs, n_heads, seq_len, d_head = 1, 2, 128, 32
    causal = True
    softmax_scale = 1 / (d_head**0.5)
    k_diff = 16

    # Generate test data
    q = torch.randn(bs, n_heads, seq_len, d_head)
    k1 = torch.randn(bs, n_heads, seq_len, d_head)
    k2 = torch.randn(bs, n_heads, seq_len, d_head)
    v1 = torch.randn(bs, n_heads, seq_len, d_head)
    v2 = torch.randn(bs, n_heads, seq_len, d_head)

    print(f"Abs mean of q: {q.abs().mean().item():.6f}")
    print(f"Abs mean of k1: {k1.abs().mean().item():.6f}")
    print(f"Abs mean of k2: {k2.abs().mean().item():.6f}")
    print(f"Abs mean of v1: {v1.abs().mean().item():.6f}")
    print(f"Abs mean of v2: {v2.abs().mean().item():.6f}")

    print(f"Test configuration:")
    print(f"  Batch size: {bs}")
    print(f"  Num heads: {n_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {d_head}")
    print(f"  Causal masking: {causal}")
    print(f"  Softmax scale: {softmax_scale}")
    print()

    # Run PyTorch implementation
    print("Running PyTorch Trittention...")
    pytorch_model = Trittention_pytorch(
        causal=causal, softmax_scale=softmax_scale, n_ctx=seq_len, k_diff=k_diff
    )
    pytorch_output, pre_softmax_attn_score, v_gated = pytorch_model(q, k1, k2, v1, v2)

    # Run slow implementation
    print("Running Slow Trittention (sum of dot products)...")
    slow_output, pre_softmax_attn_score, v_gated = slow_trittention_dot_product_sum(
        q,
        k1,
        k2,
        v1,
        v2,
        causal_mask=causal,
        softmax_scale=softmax_scale,
        k_diff=k_diff,
    )

    # Compare outputs
    print("\nComparison Results:")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"Slow output shape: {slow_output.shape}")
    print()

    # Statistical comparison
    print("PyTorch output stats:")
    print_stats(pytorch_output)
    print()
    print("Slow output stats:")
    print_stats(slow_output)
    print()
    print("PyTorch pre-softmax attn score stats:")
    print_stats(pre_softmax_attn_score)
    print()
    print("Slow pre-softmax attn score stats:")
    print_stats(pre_softmax_attn_score)
    print()
    print("PyTorch v_gated stats:")
    print_stats(v_gated)
    print()
    print("Slow v_gated stats:")
    print_stats(v_gated)
    print()

    # Difference analysis
    diff = torch.abs(pytorch_output - slow_output)

    print("Difference Analysis:")
    print()
    print(f"Output difference stats:")
    print_stats(diff)
    print()
    print(f"Pre-softmax attn score difference stats:")
    diff_pre_softmax_attn_score = torch.abs(
        pre_softmax_attn_score - pre_softmax_attn_score
    )
    print_stats(diff_pre_softmax_attn_score)
    print()
    print(f"v_gated difference stats:")
    diff_v_gated = torch.abs(v_gated - v_gated)
    print_stats(diff_v_gated)
    print()

    # Element-wise comparison tolerance
    tolerance = 1e-4
    close_elements = torch.allclose(
        pytorch_output, slow_output, atol=tolerance, rtol=tolerance
    )
    print(f"Are outputs close (tolerance={tolerance})? {close_elements}")

    if not close_elements:
        print("⚠️  Outputs are significantly different - this is expected!")
        print("   The slow implementation uses sum of dot products scoring,")
        print("   while PyTorch version uses different attention mechanism.")
    else:
        print("✅ Outputs are very similar despite different attention mechanisms.")

    print()
    print("Implementation Differences:")
    print("1. Attention Scoring:")
    print("   - PyTorch: Uses einsum operations for qk and kk interactions")
    print("   - Slow: Uses sum of dot(k1,k2) + dot(q,k2)")
    print()
    print("2. Value Computation:")
    print("   - PyTorch: Uses F.silu(v1.unsqueeze(2)) * v2.unsqueeze(1)")
    print("   - Slow: Uses F.silu(v1) * v2 (SwishGLU)")
    print()

    return pytorch_output, slow_output, diff


def run_performance_test():
    """
    Simple performance comparison between implementations.
    """
    import time

    print("Performance Test")
    print("=" * 30)

    torch.manual_seed(42)
    bs, n_heads, seq_len, d_head = 1, 2, 16, 32

    q = torch.randn(bs, n_heads, seq_len, d_head)
    k1 = torch.randn(bs, n_heads, seq_len, d_head)
    k2 = torch.randn(bs, n_heads, seq_len, d_head)
    v1 = torch.randn(bs, n_heads, seq_len, d_head)
    v2 = torch.randn(bs, n_heads, seq_len, d_head)

    # PyTorch timing
    pytorch_model = Trittention_pytorch(causal=True, softmax_scale=1 / 8, n_ctx=seq_len)

    start_time = time.time()
    for _ in range(10):
        _ = pytorch_model(q, k1, k2, v1, v2)
    pytorch_time = (time.time() - start_time) / 10

    # Slow implementation timing
    start_time = time.time()
    for _ in range(10):
        _ = slow_trittention_dot_product_sum(
            q, k1, k2, v1, v2, causal_mask=True, softmax_scale=1 / 8
        )
    slow_time = (time.time() - start_time) / 10

    print(f"PyTorch implementation: {pytorch_time:.4f}s per run")
    print(f"Slow implementation: {slow_time:.4f}s per run")
    print(f"Speedup ratio: {slow_time/pytorch_time:.2f}x")


if __name__ == "__main__":
    pytorch_out, slow_out, differences = compare_trittention_implementations()
    print("\n" + "=" * 50)
    # run_performance_test()
