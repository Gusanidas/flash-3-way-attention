import triton
import triton.language as tl
import torch


@triton.jit
def _tritt_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K1,
    K2,
    V1,
    V2,
    qvk_offset,
    stride_K1_seq,
    stride_K1_dim,
    stride_K2_seq,
    stride_K2_dim,
    stride_V1_seq,
    stride_V1_dim,
    stride_V2_seq,
    stride_V2_dim,
    softmax_scale,
    block_index_q,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
):
    if STAGE == 1:
        # From 0 to the left of the diagonal (for causal attention)
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
        lo_1, hi_1 = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Diagonal block (for causal attention)
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
        lo_1, hi_1 = 0, (block_index_q + 1) * BLOCK_SIZE_Q
    else:
        # Non-causal attention (process all tokens)
        lo, hi = 0, SEQ_LEN
        lo_1, hi_1 = 0, SEQ_LEN
    
    K2_block_ptr = tl.make_block_ptr(
        base=K2 + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K2_seq, stride_K2_dim),
        offsets=(lo, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    V2_block_ptr = tl.make_block_ptr(
        base=V2 + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V2_seq, stride_V2_dim),
        offsets=(lo, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    for start_kv_2 in range(lo, hi, BLOCK_SIZE_KV):
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV)
        
        k1_hi = tl.minimum(hi_1, start_kv_2 + BLOCK_SIZE_KV)

        K2_block = tl.load(K2_block_ptr)
        V2_block = tl.load(V2_block_ptr)
        QK2_block = Q_block[:, None, :] * K2_block[None, :, :]
        QK2_block = tl.reshape(QK2_block, (BLOCK_SIZE_Q*BLOCK_SIZE_KV, HEAD_DIM))

        for start_kv_1 in range(lo_1, k1_hi, BLOCK_SIZE_KV):
            start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV)
            
            K1_block_ptr = tl.make_block_ptr(
                base=K1 + qvk_offset,
                shape=(HEAD_DIM, SEQ_LEN),
                strides=(stride_K1_dim, stride_K1_seq),
                offsets=(0, start_kv_1),
                block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
                order=(0, 1),
            )
            V1_block_ptr = tl.make_block_ptr(
                base=V1 + qvk_offset,
                shape=(SEQ_LEN, HEAD_DIM),
                strides=(stride_V1_seq, stride_V1_dim),
                offsets=(start_kv_1, 0),
                block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
                order=(1, 0),
            )
            
            K1_block = tl.load(K1_block_ptr)
            V1_block = tl.load(V1_block_ptr)

            QKK_block = tl.dot(QK2_block, K1_block)
            QKK_block = tl.reshape(QKK_block, (BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV))

            k2_absolute_positions = start_kv_2 + offs_kv  # [BLOCK_SIZE_KV]
            k1_absolute_positions = start_kv_1 + offs_kv  # [BLOCK_SIZE_KV]
            
            # Create mask for k1 <= k2 condition within blocks
            mask_k2_k1 = k2_absolute_positions[:, None] >= k1_absolute_positions[None, :]  # [BLOCK_SIZE_KV, BLOCK_SIZE_KV]
            
            # For causal attention in STAGE 2, also apply q >= k2 masking
            if STAGE == 2:
                # q >= k2 condition (each query position can only attend to previous or current k2 positions)
                mask_q_k2 = offs_q[:, None] >= k2_absolute_positions[None, :]  # [BLOCK_SIZE_Q, BLOCK_SIZE_KV]
                
                # Combine masks to enforce k1 <= k2 <= q
                mask = mask_q_k2[:, :, None] & mask_k2_k1[None, :, :]  # [BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV]
                QKK_block = QKK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            elif STAGE == 1:
                # For non-diagonal blocks, only enforce k1 <= k2 regardless of q
                mask =  mask_k2_k1[None, :, :] #& (QKK_block <= QKK_block)
                QKK_block = QKK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            else:
                # For non-causal attention, no masking is applied
                QKK_block = QKK_block * softmax_scale

            m_ij = tl.maximum(m_i, tl.max(tl.max(QKK_block, axis=2), axis=1))
            QKK_block -= m_ij[:, None, None]

            P_block = tl.math.exp(QKK_block)
            l_ij = tl.sum(tl.sum(P_block, axis=2), axis=1)

            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            O1_block = tl.dot(tl.sum(P_block, axis=1), V1_block)
            P_block = tl.sum(P_block, axis=2)
            O2_block = tl.dot(P_block, V2_block)

            O_block = O_block * alpha[:, None] + O1_block + O2_block

            m_i = m_ij

        K2_block_ptr = tl.advance(K2_block_ptr, (BLOCK_SIZE_KV, 0))
        V2_block_ptr = tl.advance(V2_block_ptr, (BLOCK_SIZE_KV, 0))

    return O_block, l_i, m_i


@triton.jit
def _tritt_fwd(
    Q,
    K1,
    K2,
    V1,
    V2,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K1_seq,
    stride_K1_dim,
    stride_K2_seq,
    stride_K2_dim,
    stride_V1_seq,
    stride_V1_dim,
    stride_V2_seq,
    stride_V2_dim,
    stride_O_seq,
    stride_O_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    Q_block = tl.load(Q_block_ptr)

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # For causal attention (STAGE=3), we split processing into two parts
    if STAGE == 1 or STAGE == 3:
        # Process blocks to the left of the diagonal (without masking)
        O_block, l_i, m_i = _tritt_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K1,
            K2,
            V1,
            V2,
            qvk_offset,
            stride_K1_seq,
            stride_K1_dim,
            stride_K2_seq,
            stride_K2_dim,
            stride_V1_seq,
            stride_V1_dim,
            stride_V2_seq,
            stride_V2_dim,
            softmax_scale,
            block_index_q,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            SEQ_LEN,
            HEAD_DIM,
            4-STAGE,  # STAGE = 1 for left of diagonal
            offs_q,
            offs_kv,
        )
        
    if STAGE == 3:
        # Process the diagonal block (with masking)
        O_block, l_i, m_i = _tritt_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K1,
            K2,
            V1,
            V2,
            qvk_offset,
            stride_K1_seq,
            stride_K1_dim,
            stride_K2_seq,
            stride_K2_dim,
            stride_V1_seq,
            stride_V1_dim,
            stride_V2_seq,
            stride_V2_dim,
            softmax_scale,
            block_index_q,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            SEQ_LEN,
            HEAD_DIM,
            2,  # STAGE = 2 for diagonal block
            offs_q,
            offs_kv,
        )

    m_i += tl.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def tritt_fwd(Q, K1, K2, V1, V2, causal, softmax_scale):
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    O = torch.empty_like(Q)
    M = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
    )

    grid = lambda args: (
        triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
        1,
    )

    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_KV = 16

    stride_Q_batch = Q.stride(0)
    stride_Q_head = Q.stride(1)
    stride_Q_seq = Q.stride(2)
    stride_Q_dim = Q.stride(3)

    stride_K1_seq = K1.stride(2)
    stride_K1_dim = K1.stride(3)

    stride_K2_seq = K2.stride(2)
    stride_K2_dim = K2.stride(3)

    stride_V1_seq = V1.stride(2)
    stride_V1_dim = V1.stride(3)

    stride_V2_seq = V2.stride(2)
    stride_V2_dim = V2.stride(3)

    stride_O_seq = O.stride(2)
    stride_O_dim = O.stride(3)

    # Use STAGE=3 for causal attention, STAGE=4 for non-causal attention
    STAGE = 3 if causal else 1

    _tritt_fwd[grid](
        Q=Q,
        K1=K1,
        K2=K2,
        V1=V1,
        V2=V2,
        softmax_scale=softmax_scale,
        M=M,
        O=O,
        stride_Q_batch=stride_Q_batch,
        stride_Q_head=stride_Q_head,
        stride_Q_seq=stride_Q_seq,
        stride_Q_dim=stride_Q_dim,
        stride_K1_seq=stride_K1_seq,
        stride_K1_dim=stride_K1_dim,
        stride_K2_seq=stride_K2_seq,
        stride_K2_dim=stride_K2_dim,
        stride_V1_seq=stride_V1_seq,
        stride_V1_dim=stride_V1_dim,
        stride_V2_seq=stride_V2_seq,
        stride_V2_dim=stride_V2_dim,
        stride_O_seq=stride_O_seq,
        stride_O_dim=stride_O_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
        STAGE=STAGE,  # Pass the STAGE parameter
    )

    return O, M


if __name__ == "__main__":
    head_dim = 64
    q = torch.randn(1, 8, 16, head_dim)
    k1 = torch.randn(1, 8, 16, head_dim)
    k2 = torch.randn(1, 8, 16, head_dim)
    v1 = torch.randn(1, 8, 16, head_dim)
    v2 = torch.randn(1, 8, 16, head_dim)
    
    # Test causal attention
    tritt_out_causal, _ = tritt_fwd(q, k1, k2, v1, v2, causal=True, softmax_scale=1.0)
    print("Causal output shape:", tritt_out_causal.shape)
    
    # Test non-causal attention
    tritt_out_noncausal, _ = tritt_fwd(q, k1, k2, v1, v2, causal=False, softmax_scale=1.0)
    print("Non-causal output shape:", tritt_out_noncausal.shape)