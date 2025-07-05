import torch
import torch.nn.functional as F
from pytorch_bwd import pytorch_bwd
from bwd_dq_kernel import tritt_bwd_dq
from trittention_pytorch import Trittention_pytorch
from bwd_preprocess import tritt_bwd_preprocess


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


def compare_dq(causal=False):
    """Compare PyTorch and Triton implementations of dQ backward computation."""

    # Set up test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 200
    head_dim = 32
    softmax_scale = 1.0 / (head_dim**0.5)
    dtype = torch.bfloat16
    device = "cuda"
    k_diff = 1024

    # Generate random inputs
    torch.manual_seed(42)
    Q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K1 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K2 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    V1 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    V2 = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dO = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )

    model = Trittention_pytorch(
        causal=causal, k_diff=k_diff, softmax_scale=softmax_scale
    )
    O, _, _, M = model(Q, K1, K2, V1, V2)

    print("=== Comparing dQ computation ===")
    print(
        f"Input shapes: Q={Q.shape}, K1={K1.shape}, K2={K2.shape}, V1={V1.shape}, V2={V2.shape}"
    )
    print(f"O={O.shape}, dO={dO.shape}, M={M.shape}")
    print()

    # PyTorch implementation
    print("Computing dQ with PyTorch...")
    dQ_pytorch, _, _, _, _, _ = pytorch_bwd(
        Q, K1, K2, V1, V2, O, dO, softmax_scale, M, causal, k_diff
    )

    # Triton kernel implementation
    print("Computing dQ with Triton kernel...")

    # Compute D (preprocessing step)
    D = torch.sum(O * dO, dim=-1)
    D2 = tritt_bwd_preprocess(O, dO)
    print(f"D shape: {D.shape}, D2 shape: {D2.shape}")
    print(f"D mean: {D.mean().item():.6f}, D2 mean: {D2.mean().item():.6f}")
    print(f"Abs diff: {torch.abs(D - D2).mean().item():.6f}")

    # Use the new tritt_bwd_dq wrapper function
    dQ_triton, dK2_triton = tritt_bwd_dq(
        Q, K1, K2, V1, V2, D, softmax_scale, dO, M, causal, k_diff
    )
    print("-=-=-=-=-=-=-=-=-=-=-+_+_+_+_+_+_+_+_+_+_+_")
    print("-=-=-=-=-=-=-=-=-=-=-+_+_+_+_+_+_+_+_+_+_+_")
    print(f"dK2_triton mean: {dK2_triton.mean().item():.6f}")
    print(f"dK2_triton std: {dK2_triton.std().item():.6f}")
    print(f"dK2_triton max: {dK2_triton.max().item():.6f}")
    print(f"dK2_triton min: {dK2_triton.min().item():.6f}")
    print(f"dK2_triton: {dK2_triton}")
    print("-=-=-=-=-=-=-=-=-=-=-+_+_+_+_+_+_+_+_+_+_+_")
    print("-=-=-=-=-=-=-=-=-=-=-+_+_+_+_+_+_+_+_+_+_+_")

    # PyTorch autograd implementation
    print("Computing dQ with PyTorch autograd...")

    # Create fresh copies of inputs for autograd computation
    Q_autograd = Q.clone().detach().requires_grad_(True)
    K1_autograd = K1.clone().detach().requires_grad_(True)
    K2_autograd = K2.clone().detach().requires_grad_(True)
    V1_autograd = V1.clone().detach().requires_grad_(True)
    V2_autograd = V2.clone().detach().requires_grad_(True)

    # Forward pass for autograd
    model_autograd = Trittention_pytorch(
        causal=causal, k_diff=k_diff, softmax_scale=softmax_scale
    )
    O_autograd, _, _, _ = model_autograd(
        Q_autograd, K1_autograd, K2_autograd, V1_autograd, V2_autograd
    )

    # Compute dQ using autograd
    dQ_autograd = torch.autograd.grad(
        outputs=O_autograd,
        inputs=Q_autograd,
        grad_outputs=dO,
        retain_graph=True,
        create_graph=False,
    )[0]

    print()
    print("=== Results ===")
    print("PyTorch dQ:")
    print_stats(dQ_pytorch)
    print()
    print("Triton dQ:")
    print_stats(dQ_triton)
    print()
    print("PyTorch Autograd dQ:")
    print_stats(dQ_autograd)
    print()

    # Calculate differences between all pairs
    abs_diff_pytorch_triton = torch.abs(dQ_pytorch - dQ_triton)
    rel_diff_pytorch_triton = abs_diff_pytorch_triton / (torch.abs(dQ_pytorch) + 1e-5)

    abs_diff_pytorch_autograd = torch.abs(dQ_pytorch - dQ_autograd)
    rel_diff_pytorch_autograd = abs_diff_pytorch_autograd / (
        torch.abs(dQ_pytorch) + 1e-8
    )

    abs_diff_triton_autograd = torch.abs(dQ_triton - dQ_autograd)
    rel_diff_triton_autograd = abs_diff_triton_autograd / (torch.abs(dQ_triton) + 1e-5)

    print("=== Differences ===")
    print("PyTorch vs Triton - Absolute difference:")
    print_stats(abs_diff_pytorch_triton)
    print("PyTorch vs Triton - Relative difference:")
    print_stats(rel_diff_pytorch_triton)
    print()

    print("PyTorch vs Autograd - Absolute difference:")
    print_stats(abs_diff_pytorch_autograd)
    print("PyTorch vs Autograd - Relative difference:")
    print_stats(rel_diff_pytorch_autograd)
    print()

    print("Triton vs Autograd - Absolute difference:")
    print_stats(abs_diff_triton_autograd)
    print("Triton vs Autograd - Relative difference:")
    print_stats(rel_diff_triton_autograd)
    print()

    # Check if outputs are close
    tolerance = 1e-3
    pytorch_triton_close = torch.allclose(
        dQ_pytorch, dQ_triton, atol=tolerance, rtol=tolerance
    )
    pytorch_autograd_close = torch.allclose(
        dQ_pytorch, dQ_autograd, atol=tolerance, rtol=tolerance
    )
    triton_autograd_close = torch.allclose(
        dQ_triton, dQ_autograd, atol=tolerance, rtol=tolerance
    )

    print(f"PyTorch vs Triton close (tolerance={tolerance}): {pytorch_triton_close}")
    print(
        f"PyTorch vs Autograd close (tolerance={tolerance}): {pytorch_autograd_close}"
    )
    print(f"Triton vs Autograd close (tolerance={tolerance}): {triton_autograd_close}")

    if not pytorch_triton_close:
        print("WARNING: PyTorch and Triton outputs differ significantly!")
        print("First few elements comparison:")
        print("PyTorch:", dQ_pytorch.flatten()[:10])
        print("Triton:", dQ_triton.flatten()[:10])

    if not pytorch_autograd_close:
        print("WARNING: PyTorch and Autograd outputs differ significantly!")
        print("First few elements comparison:")
        print("PyTorch:", dQ_pytorch.flatten()[:10])
        print("Autograd:", dQ_autograd.flatten()[:10])

    if not triton_autograd_close:
        print("WARNING: Triton and Autograd outputs differ significantly!")
        print("First few elements comparison:")
        print("Triton:", dQ_triton.flatten()[:10])
        print("Autograd:", dQ_autograd.flatten()[:10])

    if pytorch_triton_close and pytorch_autograd_close and triton_autograd_close:
        print("SUCCESS: All three dQ computations match within tolerance!")


if __name__ == "__main__":
    print("=== Testing Causal Attention ===")
    # compare_dq(causal=True)
    print("\n" + "=" * 50 + "\n")
    print("=== Testing Non-Causal Attention ===")
    compare_dq(causal=False)
