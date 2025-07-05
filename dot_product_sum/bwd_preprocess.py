import triton
import triton.language as tl
import torch


@triton.jit
def _tritt_bwd_preprocess(
    O,
    dO,
    D,
    dO_stride_S,
    dO_stride_D,
    O_stride_S,
    O_stride_D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute D = sum(O * dO) for the backward pass."""
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offset_batch_head = head_idx * SEQ_LEN * HEAD_DIM

    start_q = block_idx * BLOCK_SIZE_Q
    offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)
    offs_d = tl.arange(0, HEAD_DIM)

    # Check if we're within bounds
    mask = offs_q < SEQ_LEN

    # Load O and dO blocks
    o_ptrs = (
        O
        + offset_batch_head
        + offs_q[:, None] * O_stride_S
        + offs_d[None, :] * O_stride_D
    )
    do_ptrs = (
        dO
        + offset_batch_head
        + offs_q[:, None] * dO_stride_S
        + offs_d[None, :] * dO_stride_D
    )

    o_block = tl.load(o_ptrs, mask=mask[:, None], other=0.0)
    do_block = tl.load(do_ptrs, mask=mask[:, None], other=0.0)

    # Compute D = sum(O * dO)
    delta = tl.sum(o_block * do_block, axis=1)

    # Store the result
    d_ptrs = D + head_idx * SEQ_LEN + offs_q
    tl.store(d_ptrs, delta, mask=mask)


def tritt_bwd_preprocess(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for the backward preprocessing kernel.

    Computes D = sum(O * dO) along the head dimension for each sequence position.
    This is used in the backward pass of attention mechanisms.

    Args:
        O: Output tensor from forward pass, shape [B, H, S, D]
        dO: Gradient of output tensor, shape [B, H, S, D]

    Returns:
        D: Preprocessing result, shape [B, H, S]
    """
    # Get tensor dimensions
    batch_size, num_heads, seq_len, head_dim = O.shape

    # Validate input shapes
    assert dO.shape == O.shape, f"Shape mismatch: O={O.shape}, dO={dO.shape}"
    assert O.device == dO.device, "O and dO must be on the same device"
    assert O.dtype == dO.dtype, "O and dO must have the same dtype"

    # Allocate output tensor D with shape [B, H, S]
    D = torch.empty((batch_size, num_heads, seq_len), device=O.device, dtype=O.dtype)

    # Kernel configuration
    BLOCK_SIZE_Q = 16  # Block size for sequence dimension

    # Calculate grid dimensions: (seq_blocks, batch*heads)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_Q), batch_size * num_heads)

    # Get strides for O and dO tensors
    # O and dO have shape [B, H, S, D], so strides are in same order
    O_stride_S = O.stride(2)  # stride along sequence dimension
    O_stride_D = O.stride(3)  # stride along head dimension
    dO_stride_S = dO.stride(2)  # stride along sequence dimension
    dO_stride_D = dO.stride(3)  # stride along head dimension

    # Launch the Triton kernel
    _tritt_bwd_preprocess[grid](
        O,  # Output tensor
        dO,  # Gradient of output
        D,  # Result tensor to fill
        dO_stride_S,  # dO stride along sequence
        dO_stride_D,  # dO stride along head dimension
        O_stride_S,  # O stride along sequence
        O_stride_D,  # O stride along head dimension
        seq_len,  # Sequence length
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,  # Block size (compile-time constant)
        HEAD_DIM=head_dim,  # Head dimension (compile-time constant)
    )

    return D


def pytorch_bwd_preprocess(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of the backward preprocessing.

    Computes D = sum(O * dO) along the head dimension for each sequence position.
    This is equivalent to the Triton kernel but uses PyTorch operations.

    Args:
        O: Output tensor from forward pass, shape [B, H, S, D]
        dO: Gradient of output tensor, shape [B, H, S, D]

    Returns:
        D: Preprocessing result, shape [B, H, S]
    """
    # Validate input shapes
    assert dO.shape == O.shape, f"Shape mismatch: O={O.shape}, dO={dO.shape}"
    assert O.device == dO.device, "O and dO must be on the same device"
    assert O.dtype == dO.dtype, "O and dO must have the same dtype"

    # Compute D = sum(O * dO) along the head dimension (dim=-1)
    D = torch.sum(O * dO, dim=-1)

    return D


if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64

    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    O = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    dO = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )

    print(f"Testing with tensors of shape: {O.shape}")
    print(f"Device: {device}")

    # Run PyTorch implementation
    D_pytorch = pytorch_bwd_preprocess(O, dO)
    print(f"PyTorch result shape: {D_pytorch.shape}")

    # Run Triton implementation (only if CUDA is available)
    if device.type == "cuda":
        try:
            D_triton = tritt_bwd_preprocess(O, dO)
            print(f"Triton result shape: {D_triton.shape}")

            # Compare results
            max_diff = torch.max(torch.abs(D_pytorch - D_triton)).item()
            print(f"Maximum difference between implementations: {max_diff}")

            # Check if results are close
            are_close = torch.allclose(D_pytorch, D_triton, rtol=1e-5, atol=1e-8)
            print(f"Results are close (rtol=1e-5, atol=1e-8): {are_close}")

        except Exception as e:
            print(f"Triton implementation failed: {e}")
            print("This is expected if Triton is not installed or not available.")
    else:
        print("Skipping Triton implementation (CUDA not available)")

    print("\nComparison completed!")
