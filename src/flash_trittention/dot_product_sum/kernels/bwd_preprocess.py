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

    mask = offs_q < SEQ_LEN

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

    delta = tl.sum(o_block * do_block, axis=1)

    d_ptrs = D + head_idx * SEQ_LEN + offs_q
    tl.store(d_ptrs, delta, mask=mask)


def tritt_bwd_preprocess(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for the backward preprocessing kernel.

    Computes D = sum(O * dO) along the head dimension for each sequence position.
    """
    batch_size, num_heads, seq_len, head_dim = O.shape

    assert dO.shape == O.shape, f"Shape mismatch: O={O.shape}, dO={dO.shape}"
    assert O.device == dO.device, "O and dO must be on the same device"
    assert O.dtype == dO.dtype, "O and dO must have the same dtype"

    D = torch.empty((batch_size, num_heads, seq_len), device=O.device, dtype=O.dtype)

    BLOCK_SIZE_Q = 16  # Block size for sequence dimension

    grid = (triton.cdiv(seq_len, BLOCK_SIZE_Q), batch_size * num_heads)

    O_stride_S = O.stride(2)
    O_stride_D = O.stride(3)
    dO_stride_S = dO.stride(2)
    dO_stride_D = dO.stride(3)

    _tritt_bwd_preprocess[grid](
        O,
        dO,
        D,
        dO_stride_S,
        dO_stride_D,
        O_stride_S,
        O_stride_D,
        seq_len,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        HEAD_DIM=head_dim,
    )

    return D


def pytorch_bwd_preprocess(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of the backward preprocessing.

    Computes D = sum(O * dO) along the head dimension for each sequence position.
    This is equivalent to the Triton kernel but uses PyTorch operations.
    """
    assert dO.shape == O.shape, f"Shape mismatch: O={O.shape}, dO={dO.shape}"
    assert O.device == dO.device, "O and dO must be on the same device"
    assert O.dtype == dO.dtype, "O and dO must have the same dtype"

    D = torch.sum(O * dO, dim=-1)
    return D
