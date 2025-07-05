@triton.jit
def _tritt_bwd_preprocess(
    O,
    dO,
    D,
    dO_stride_S, dO_stride_D,
    O_stride_S, O_stride_D,
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
    o_ptrs = O + offset_batch_head + offs_q[:, None] * O_stride_S + offs_d[None, :] * O_stride_D
    do_ptrs = dO + offset_batch_head + offs_q[:, None] * dO_stride_S + offs_d[None, :] * dO_stride_D
    
    o_block = tl.load(o_ptrs, mask=mask[:, None], other=0.0)
    do_block = tl.load(do_ptrs, mask=mask[:, None], other=0.0)
    
    # Compute D = sum(O * dO)
    delta = tl.sum(o_block * do_block, axis=1)
    
    # Store the result
    d_ptrs = D + head_idx * SEQ_LEN + offs_q
    tl.store(d_ptrs, delta, mask=mask)