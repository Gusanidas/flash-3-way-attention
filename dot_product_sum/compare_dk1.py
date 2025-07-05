import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_bwd import pytorch_bwd
from bwd_dk1_dv1_kernel import tritt_bwd_dk1_dv1
from trittention_pytorch import Trittention_pytorch

# Test parameters
BATCH_SIZE = 2
NUM_HEADS = 4
SEQ_LEN = 128
HEAD_DIM = 32
SOFTMAX_SCALE = 1.0 / (HEAD_DIM**0.5)
DTYPE = torch.float32
DEVICE = "cuda"
K_DIFF = 1024

# Plotting control
PLOT_K1 = True


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


def plot_matrix(matrix, title="Matrix", max_size=50, cmap="RdBu_r", save_path=None):
    """Plot a matrix as a colorful heatmap with optional truncation."""
    print(f"\n{title} (shape: {matrix.shape}):")

    # Convert to numpy for plotting
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = matrix

    rows, cols = matrix_np.shape

    # Truncate if matrix is too large
    if rows > max_size or cols > max_size:
        show_rows = min(max_size, rows)
        show_cols = min(max_size, cols)
        matrix_np = matrix_np[:show_rows, :show_cols]
        title += f" (truncated to {show_rows}x{show_cols})"

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(matrix_np, cmap=cmap, aspect="auto")

    # Add colorbar
    plt.colorbar(im, label="Value")

    # Set title and labels
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    # Add grid for better readability (only for smaller matrices)
    if matrix_np.shape[0] <= 20 and matrix_np.shape[1] <= 20:
        plt.grid(True, alpha=0.3)
        # Set tick marks for each cell
        plt.xticks(range(matrix_np.shape[1]))
        plt.yticks(range(matrix_np.shape[0]))

    # Add text annotations for very small matrices
    if matrix_np.shape[0] <= 10 and matrix_np.shape[1] <= 10:
        for i in range(matrix_np.shape[0]):
            for j in range(matrix_np.shape[1]):
                text = plt.text(
                    j,
                    i,
                    f"{matrix_np[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show the plot
    plt.show()

    # Print basic statistics
    print(
        f"Matrix statistics - Mean: {matrix_np.mean():.6f}, Std: {matrix_np.std():.6f}"
    )
    print(f"Min: {matrix_np.min():.6f}, Max: {matrix_np.max():.6f}")
    print()


def compare_dk1_dv1(causal=True):
    """Compare PyTorch and Triton implementations of dK1, dV1, and dV2 backward computation."""

    # Use global test parameters
    batch_size = BATCH_SIZE
    num_heads = NUM_HEADS
    seq_len = SEQ_LEN
    head_dim = HEAD_DIM
    softmax_scale = SOFTMAX_SCALE
    dtype = DTYPE
    device = DEVICE
    k_diff = K_DIFF

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

    # Get O and M from forward pass
    model = Trittention_pytorch(
        causal=causal, k_diff=k_diff, softmax_scale=softmax_scale
    )
    O, _, _, M = model(Q, K1, K2, V1, V2)

    print("=== Comparing dK1, dV1, and dV2 computation ===")
    print(
        f"Input shapes: Q={Q.shape}, K1={K1.shape}, K2={K2.shape}, V1={V1.shape}, V2={V2.shape}"
    )
    print(f"O={O.shape}, dO={dO.shape}, M={M.shape}")
    print(f"Causal: {causal}, k_diff: {k_diff}")
    print()

    # PyTorch implementation
    print("Computing gradients with PyTorch...")
    _, dK1_pytorch, _, dV1_pytorch, dV2_pytorch, _ = pytorch_bwd(
        Q, K1, K2, V1, V2, O, dO, softmax_scale, M, causal, k_diff
    )

    # Triton kernel implementation
    print("Computing gradients with Triton kernel...")

    # Compute D (preprocessing step)
    D = torch.sum(O * dO, dim=-1)

    # Use the new tritt_bwd_dk1_dv1 function that computes dK1, dV1, and dV2
    dK1_triton, dV1_triton, dV2_triton = tritt_bwd_dk1_dv1(
        Q, K1, K2, V1, V2, D, softmax_scale, dO, M, causal, k_diff
    )

    # PyTorch autograd implementation
    print("Computing dK1, dV1, and dV2 with PyTorch autograd...")

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

    # Compute dK1, dV1, and dV2 using autograd
    dK1_autograd = torch.autograd.grad(
        outputs=O_autograd,
        inputs=K1_autograd,
        grad_outputs=dO,
        retain_graph=True,
        create_graph=False,
    )[0]

    dV1_autograd = torch.autograd.grad(
        outputs=O_autograd,
        inputs=V1_autograd,
        grad_outputs=dO,
        retain_graph=True,
        create_graph=False,
    )[0]

    dV2_autograd = torch.autograd.grad(
        outputs=O_autograd,
        inputs=V2_autograd,
        grad_outputs=dO,
        retain_graph=True,
        create_graph=False,
    )[0]

    print()
    print("=== dK1 Results ===")
    print("PyTorch dK1:")
    print_stats(dK1_pytorch)
    print()
    print("Triton dK1:")
    print_stats(dK1_triton)
    print()
    print("PyTorch Autograd dK1:")
    print_stats(dK1_autograd)
    print()

    # Calculate dK1 differences between all pairs
    abs_diff_k1_pytorch_triton = torch.abs(dK1_pytorch - dK1_triton)
    rel_diff_k1_pytorch_triton = abs_diff_k1_pytorch_triton / (
        torch.abs(dK1_pytorch) + 1e-8
    )

    abs_diff_k1_pytorch_autograd = torch.abs(dK1_pytorch - dK1_autograd)
    rel_diff_k1_pytorch_autograd = abs_diff_k1_pytorch_autograd / (
        torch.abs(dK1_pytorch) + 1e-8
    )

    abs_diff_k1_triton_autograd = torch.abs(dK1_triton - dK1_autograd)
    rel_diff_k1_triton_autograd = abs_diff_k1_triton_autograd / (
        torch.abs(dK1_triton) + 1e-8
    )

    print("=== dK1 Differences ===")
    print("PyTorch vs Triton - Absolute difference:")
    print_stats(abs_diff_k1_pytorch_triton)
    print("PyTorch vs Triton - Relative difference:")
    print_stats(rel_diff_k1_pytorch_triton)
    print()

    print("PyTorch vs Autograd - Absolute difference:")
    print_stats(abs_diff_k1_pytorch_autograd)
    print("PyTorch vs Autograd - Relative difference:")
    print_stats(rel_diff_k1_pytorch_autograd)
    print()

    print("Triton vs Autograd - Absolute difference:")
    print_stats(abs_diff_k1_triton_autograd)
    print("Triton vs Autograd - Relative difference:")
    print_stats(rel_diff_k1_triton_autograd)
    print()

    # Check if dK1 outputs are close
    tolerance = 1e-3
    are_close_k1_pytorch_triton = torch.allclose(
        dK1_pytorch, dK1_triton, atol=tolerance, rtol=tolerance
    )
    are_close_k1_pytorch_autograd = torch.allclose(
        dK1_pytorch, dK1_autograd, atol=tolerance, rtol=tolerance
    )
    are_close_k1_triton_autograd = torch.allclose(
        dK1_triton, dK1_autograd, atol=tolerance, rtol=tolerance
    )

    print(
        f"dK1 PyTorch vs Triton close (tolerance={tolerance}): {are_close_k1_pytorch_triton}"
    )
    print(
        f"dK1 PyTorch vs Autograd close (tolerance={tolerance}): {are_close_k1_pytorch_autograd}"
    )
    print(
        f"dK1 Triton vs Autograd close (tolerance={tolerance}): {are_close_k1_triton_autograd}"
    )

    if not are_close_k1_pytorch_triton:
        print("WARNING: dK1 PyTorch and Triton outputs differ significantly!")
        print("First few dK1 elements comparison:")
        print("PyTorch:", dK1_pytorch.flatten()[:10])
        print("Triton:", dK1_triton.flatten()[:10])

    if not are_close_k1_pytorch_autograd:
        print("WARNING: dK1 PyTorch and Autograd outputs differ significantly!")
        print("First few dK1 elements comparison:")
        print("PyTorch:", dK1_pytorch.flatten()[:10])
        print("Autograd:", dK1_autograd.flatten()[:10])

    if not are_close_k1_triton_autograd:
        print("WARNING: dK1 Triton and Autograd outputs differ significantly!")
        print("First few dK1 elements comparison:")
        print("Triton:", dK1_triton.flatten()[:10])
        print("Autograd:", dK1_autograd.flatten()[:10])

    if (
        are_close_k1_pytorch_triton
        and are_close_k1_pytorch_autograd
        and are_close_k1_triton_autograd
    ):
        print("SUCCESS: All three dK1 computations match within tolerance!")

    # Plot matrices for batch=0, head=0 (if enabled)
    if PLOT_K1:
        plot_matrix(dK1_pytorch[0, 0], "PyTorch dK1 [batch=0, head=0]", max_size=32)
        plot_matrix(dK1_triton[0, 0], "Triton dK1 [batch=0, head=0]", max_size=32)

        # Plot difference matrix
        diff_matrix = dK1_pytorch[0, 0] - dK1_triton[0, 0]
        plot_matrix(
            diff_matrix,
            "Difference dK1 [batch=0, head=0] (PyTorch - Triton)",
            max_size=32,
            cmap="seismic",
        )

    print()
    print("=== dV1 Results ===")
    print("PyTorch dV1:")
    print_stats(dV1_pytorch)
    print()
    print("Triton dV1:")
    print_stats(dV1_triton)
    print()
    print("PyTorch Autograd dV1:")
    print_stats(dV1_autograd)
    print()

    # Calculate dV1 differences between all pairs
    abs_diff_v1_pytorch_triton = torch.abs(dV1_pytorch - dV1_triton)
    rel_diff_v1_pytorch_triton = abs_diff_v1_pytorch_triton / (
        torch.abs(dV1_pytorch) + 1e-8
    )

    abs_diff_v1_pytorch_autograd = torch.abs(dV1_pytorch - dV1_autograd)
    rel_diff_v1_pytorch_autograd = abs_diff_v1_pytorch_autograd / (
        torch.abs(dV1_pytorch) + 1e-8
    )

    abs_diff_v1_triton_autograd = torch.abs(dV1_triton - dV1_autograd)
    rel_diff_v1_triton_autograd = abs_diff_v1_triton_autograd / (
        torch.abs(dV1_triton) + 1e-8
    )

    print("=== dV1 Differences ===")
    print("PyTorch vs Triton - Absolute difference:")
    print_stats(abs_diff_v1_pytorch_triton)
    print("PyTorch vs Triton - Relative difference:")
    print_stats(rel_diff_v1_pytorch_triton)
    print()

    print("PyTorch vs Autograd - Absolute difference:")
    print_stats(abs_diff_v1_pytorch_autograd)
    print("PyTorch vs Autograd - Relative difference:")
    print_stats(rel_diff_v1_pytorch_autograd)
    print()

    print("Triton vs Autograd - Absolute difference:")
    print_stats(abs_diff_v1_triton_autograd)
    print("Triton vs Autograd - Relative difference:")
    print_stats(rel_diff_v1_triton_autograd)
    print()

    # Check if dV1 outputs are close
    are_close_v1_pytorch_triton = torch.allclose(
        dV1_pytorch, dV1_triton, atol=tolerance, rtol=tolerance
    )
    are_close_v1_pytorch_autograd = torch.allclose(
        dV1_pytorch, dV1_autograd, atol=tolerance, rtol=tolerance
    )
    are_close_v1_triton_autograd = torch.allclose(
        dV1_triton, dV1_autograd, atol=tolerance, rtol=tolerance
    )

    print(
        f"dV1 PyTorch vs Triton close (tolerance={tolerance}): {are_close_v1_pytorch_triton}"
    )
    print(
        f"dV1 PyTorch vs Autograd close (tolerance={tolerance}): {are_close_v1_pytorch_autograd}"
    )
    print(
        f"dV1 Triton vs Autograd close (tolerance={tolerance}): {are_close_v1_triton_autograd}"
    )

    if not are_close_v1_pytorch_triton:
        print("WARNING: dV1 PyTorch and Triton outputs differ significantly!")
        print("First few dV1 elements comparison:")
        print("PyTorch:", dV1_pytorch.flatten()[:10])
        print("Triton:", dV1_triton.flatten()[:10])

    if not are_close_v1_pytorch_autograd:
        print("WARNING: dV1 PyTorch and Autograd outputs differ significantly!")
        print("First few dV1 elements comparison:")
        print("PyTorch:", dV1_pytorch.flatten()[:10])
        print("Autograd:", dV1_autograd.flatten()[:10])

    if not are_close_v1_triton_autograd:
        print("WARNING: dV1 Triton and Autograd outputs differ significantly!")
        print("First few dV1 elements comparison:")
        print("Triton:", dV1_triton.flatten()[:10])
        print("Autograd:", dV1_autograd.flatten()[:10])

    if (
        are_close_v1_pytorch_triton
        and are_close_v1_pytorch_autograd
        and are_close_v1_triton_autograd
    ):
        print("SUCCESS: All three dV1 computations match within tolerance!")

    print()
    print("=== dV2 Results ===")
    print("PyTorch dV2:")
    print_stats(dV2_pytorch)
    print()
    print("Triton dV2:")
    print_stats(dV2_triton)
    print()
    print("PyTorch Autograd dV2:")
    print_stats(dV2_autograd)
    print()

    # Calculate dV2 differences between all pairs
    abs_diff_v2_pytorch_triton = torch.abs(dV2_pytorch - dV2_triton)
    rel_diff_v2_pytorch_triton = abs_diff_v2_pytorch_triton / (
        torch.abs(dV2_pytorch) + 1e-8
    )

    abs_diff_v2_pytorch_autograd = torch.abs(dV2_pytorch - dV2_autograd)
    rel_diff_v2_pytorch_autograd = abs_diff_v2_pytorch_autograd / (
        torch.abs(dV2_pytorch) + 1e-8
    )

    abs_diff_v2_triton_autograd = torch.abs(dV2_triton - dV2_autograd)
    rel_diff_v2_triton_autograd = abs_diff_v2_triton_autograd / (
        torch.abs(dV2_triton) + 1e-8
    )

    print("=== dV2 Differences ===")
    print("PyTorch vs Triton - Absolute difference:")
    print_stats(abs_diff_v2_pytorch_triton)
    print("PyTorch vs Triton - Relative difference:")
    print_stats(rel_diff_v2_pytorch_triton)
    print()

    print("PyTorch vs Autograd - Absolute difference:")
    print_stats(abs_diff_v2_pytorch_autograd)
    print("PyTorch vs Autograd - Relative difference:")
    print_stats(rel_diff_v2_pytorch_autograd)
    print()

    print("Triton vs Autograd - Absolute difference:")
    print_stats(abs_diff_v2_triton_autograd)
    print("Triton vs Autograd - Relative difference:")
    print_stats(rel_diff_v2_triton_autograd)
    print()

    # Check if dV2 outputs are close
    are_close_v2_pytorch_triton = torch.allclose(
        dV2_pytorch, dV2_triton, atol=tolerance, rtol=tolerance
    )
    are_close_v2_pytorch_autograd = torch.allclose(
        dV2_pytorch, dV2_autograd, atol=tolerance, rtol=tolerance
    )
    are_close_v2_triton_autograd = torch.allclose(
        dV2_triton, dV2_autograd, atol=tolerance, rtol=tolerance
    )

    print(
        f"dV2 PyTorch vs Triton close (tolerance={tolerance}): {are_close_v2_pytorch_triton}"
    )
    print(
        f"dV2 PyTorch vs Autograd close (tolerance={tolerance}): {are_close_v2_pytorch_autograd}"
    )
    print(
        f"dV2 Triton vs Autograd close (tolerance={tolerance}): {are_close_v2_triton_autograd}"
    )

    if not are_close_v2_pytorch_triton:
        print("WARNING: dV2 PyTorch and Triton outputs differ significantly!")
        print("First few dV2 elements comparison:")
        print("PyTorch:", dV2_pytorch.flatten()[:10])
        print("Triton:", dV2_triton.flatten()[:10])

    if not are_close_v2_pytorch_autograd:
        print("WARNING: dV2 PyTorch and Autograd outputs differ significantly!")
        print("First few dV2 elements comparison:")
        print("PyTorch:", dV2_pytorch.flatten()[:10])
        print("Autograd:", dV2_autograd.flatten()[:10])

    if not are_close_v2_triton_autograd:
        print("WARNING: dV2 Triton and Autograd outputs differ significantly!")
        print("First few dV2 elements comparison:")
        print("Triton:", dV2_triton.flatten()[:10])
        print("Autograd:", dV2_autograd.flatten()[:10])

    if (
        are_close_v2_pytorch_triton
        and are_close_v2_pytorch_autograd
        and are_close_v2_triton_autograd
    ):
        print("SUCCESS: All three dV2 computations match within tolerance!")

    print()
    print("=== Overall Summary ===")
    all_k1_close = (
        are_close_k1_pytorch_triton
        and are_close_k1_pytorch_autograd
        and are_close_k1_triton_autograd
    )
    all_v1_close = (
        are_close_v1_pytorch_triton
        and are_close_v1_pytorch_autograd
        and are_close_v1_triton_autograd
    )
    all_v2_close = (
        are_close_v2_pytorch_triton
        and are_close_v2_pytorch_autograd
        and are_close_v2_triton_autograd
    )
    all_close = all_k1_close and all_v1_close and all_v2_close

    print(f"All dK1 computations match: {all_k1_close}")
    print(f"All dV1 computations match: {all_v1_close}")
    print(f"All dV2 computations match: {all_v2_close}")
    print(f"All gradient computations match: {all_close}")

    if all_close:
        print(
            "🎉 SUCCESS: All gradient computations (PyTorch, Triton, Autograd) match within tolerance!"
        )
    else:
        failed_gradients = []
        if not all_k1_close:
            failed_gradients.append("dK1")
        if not all_v1_close:
            failed_gradients.append("dV1")
        if not all_v2_close:
            failed_gradients.append("dV2")
        print(
            f"❌ FAILED: {', '.join(failed_gradients)} gradient(s) differ significantly between implementations!"
        )

    return (
        (dK1_pytorch, dK1_triton, dK1_autograd),
        (dV1_pytorch, dV1_triton, dV1_autograd),
        (dV2_pytorch, dV2_triton, dV2_autograd),
    )


def compare_dk1(causal=True):
    """Legacy function name for backward compatibility - now compares dK1, dV1, and dV2."""
    print("NOTE: compare_dk1 now compares dK1, dV1, and dV2 gradients.")
    print("Consider using compare_dk1_dv1 for clarity.")
    print()
    return compare_dk1_dv1(causal)


if __name__ == "__main__":
    print("=== Testing Causal Attention ===")
    compare_dk1_dv1(causal=True)
    print("\n" + "=" * 70 + "\n")
    print("=== Testing Non-Causal Attention ===")
    compare_dk1_dv1(causal=False)
