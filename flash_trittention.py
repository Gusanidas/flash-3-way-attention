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
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    lo, hi = 0, SEQ_LEN

    # Create block pointers inside the inner function

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
        # start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K1_block_ptr = tl.make_block_ptr(
            base=K1 + qvk_offset,
            shape=(HEAD_DIM, SEQ_LEN),
            strides=(stride_K1_dim, stride_K1_seq),
            offsets=(0, lo),
            block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
            order=(0, 1),
        )
        V1_block_ptr = tl.make_block_ptr(
            base=V1 + qvk_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_V1_seq, stride_V1_dim),
            offsets=(lo, 0),
            block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
            order=(1, 0),
        )

        K2_block = tl.load(K2_block_ptr)
        V2_block = tl.load(V2_block_ptr)
        QK2_block = Q_block[:, None, :] * K2_block[None, :, :]
        QK2_block = tl.reshape(QK2_block, (BLOCK_SIZE_KV*BLOCK_SIZE_KV, HEAD_DIM))

        for start_kv_1 in range(lo, hi, BLOCK_SIZE_KV2):
            K1_block = tl.load(K1_block_ptr)
            V1_block = tl.load(V1_block_ptr)

            #QKK_block = tl.sum(Q_block[:,None, None,:] * K1_block[None, :, None,:] * K2_block[None, None, :,:], axis=3)
            QKK_block = tl.dot(QK2_block, K1_block)
            QKK_block = tl.reshape(QKK_block, (BLOCK_SIZE_KV, BLOCK_SIZE_KV, BLOCK_SIZE_KV))
            m_ij = tl.maximum(m_i, tl.max(tl.max(QKK_block, axis=2), axis=1))
            QKK_block = QKK_block * softmax_scale - m_ij[:, None, None]

            P_block = tl.math.exp(QKK_block)
            l_ij = tl.sum(tl.sum(P_block, axis=2), axis=1)

            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            O1_block = tl.dot(tl.sum(P_block, axis=1), V2_block)
            P_block = tl.sum(P_block, axis=2)
            O2_block = tl.dot(P_block, V1_block)

            O_block = O_block * alpha[:, None] + O1_block + O2_block

            m_i = m_ij

            V1_block_ptr = tl.advance(V1_block_ptr, (BLOCK_SIZE_KV, 0))
            K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_SIZE_KV))

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
    stride_K1_batch,
    stride_K1_head,
    stride_K1_seq,
    stride_K1_dim,
    stride_K2_batch,
    stride_K2_head,
    stride_K2_seq,
    stride_K2_dim,
    stride_V1_batch,
    stride_V1_head,
    stride_V1_seq,
    stride_V1_dim,
    stride_V2_batch,
    stride_V2_head,
    stride_V2_seq,
    stride_V2_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
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

    STAGE = 3
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
        BLOCK_SIZE_KV,
        BLOCK_SIZE_KV2,
        SEQ_LEN,
        HEAD_DIM,
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
    BLOCK_SIZE_KV2 = 16

    stride_Q_batch = Q.stride(0)
    stride_Q_head = Q.stride(1)
    stride_Q_seq = Q.stride(2)
    stride_Q_dim = Q.stride(3)

    stride_K1_batch = K1.stride(0)
    stride_K1_head = K1.stride(1)
    stride_K1_seq = K1.stride(2)
    stride_K1_dim = K1.stride(3)

    stride_K2_batch = K2.stride(0)
    stride_K2_head = K2.stride(1)
    stride_K2_seq = K2.stride(2)
    stride_K2_dim = K2.stride(3)

    stride_V1_batch = V1.stride(0)
    stride_V1_head = V1.stride(1)
    stride_V1_seq = V1.stride(2)
    stride_V1_dim = V1.stride(3)

    stride_V2_batch = V2.stride(0)
    stride_V2_head = V2.stride(1)
    stride_V2_seq = V2.stride(2)
    stride_V2_dim = V2.stride(3)

    stride_O_batch = O.stride(0)
    stride_O_head = O.stride(1)
    stride_O_seq = O.stride(2)
    stride_O_dim = O.stride(3)

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
        stride_K1_batch=stride_K1_batch,
        stride_K1_head=stride_K1_head,
        stride_K1_seq=stride_K1_seq,
        stride_K1_dim=stride_K1_dim,
        stride_K2_batch=stride_K2_batch,
        stride_K2_head=stride_K2_head,
        stride_K2_seq=stride_K2_seq,
        stride_K2_dim=stride_K2_dim,
        stride_V1_batch=stride_V1_batch,
        stride_V1_head=stride_V1_head,
        stride_V1_seq=stride_V1_seq,
        stride_V1_dim=stride_V1_dim,
        stride_V2_batch=stride_V2_batch,
        stride_V2_head=stride_V2_head,
        stride_V2_seq=stride_V2_seq,
        stride_V2_dim=stride_V2_dim,
        stride_O_batch=stride_O_batch,
        stride_O_head=stride_O_head,
        stride_O_seq=stride_O_seq,
        stride_O_dim=stride_O_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        BLOCK_SIZE_KV2=BLOCK_SIZE_KV2,
        BATCH_SIZE=BATCH_SIZE,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
    )

    return O, M


if __name__ == "__main__":
    head_dim = 64
    q = torch.randn(1, 8, 16, head_dim)
    k1 = torch.randn(1, 8, 16, head_dim)
    k2 = torch.randn(1, 8, 16, head_dim)
    v1 = torch.randn(1, 8, 16, head_dim)
    v2 = torch.randn(1, 8, 16, head_dim)
    tritt_out_flash, _ = tritt_fwd(q, k1, k2, v1, v2, causal=True, softmax_scale=1.0)
    print(tritt_out_flash.shape)

