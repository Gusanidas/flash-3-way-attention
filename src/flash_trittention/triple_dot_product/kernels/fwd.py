import triton
import triton.language as tl
from .intervals import interval_k1, interval_k2_from_q
from .utils import get_autotune_config, get_causal_mask


@triton.autotune(configs=get_autotune_config(), key=["HEAD_DIM", "SEQ_LEN"])
@triton.jit
def _tritt_fwd(
    Q,
    K1,
    K2,
    V1,
    V2,
    softmax_scale,
    M,
    L,
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
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_pow2: tl.constexpr,
    kk_left: tl.constexpr,
    kk_right: tl.constexpr,
    k1_window: tl.constexpr,
    k2_window: tl.constexpr,
    causal: tl.constexpr,
    convert_to_float32: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # Direct pointer arithmetic for Q and O (no block pointers)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    q_block_ptr = (
        Q
        + qvk_offset
        + offs_q[:, None] * stride_Q_seq
        + tl.arange(0, HEAD_DIM_pow2)[None, :] * stride_Q_dim
    )
    mask_q = offs_q < SEQ_LEN
    mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM
    Q_block = tl.load(
        q_block_ptr, mask=mask_q[:, None] & mask_head_dim[None, :], other=0.0
    )
    softmax_scale = softmax_scale.to(Q_block.dtype)
    Q_block = Q_block * softmax_scale
    if convert_to_float32:
        Q_block = Q_block.to(tl.float32)

    o_block_ptr = (
        O
        + qvk_offset
        + offs_q[:, None] * stride_O_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_O_dim
    )

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM_pow2], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - 1000
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    lo_k2, hi_k2 = interval_k2_from_q(
        causal=causal,
        seq_len=SEQ_LEN,
        start_q=block_index_q * BLOCK_SIZE_Q,
        block_q=BLOCK_SIZE_Q,
        block_k=BLOCK_SIZE_KV,
        k2_window=k2_window,
    )
    mask_q = offs_q < SEQ_LEN

    for start_kv_2 in range(lo_k2, hi_k2, BLOCK_SIZE_KV):
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV)

        kv2_seq_offs = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM
        mask_kv2 = kv2_seq_offs < SEQ_LEN
        k2_block_ptr = (
            K2
            + qvk_offset
            + tl.arange(0, HEAD_DIM)[:, None] * stride_K2_dim
            + kv2_seq_offs[None, :] * stride_K2_seq
        )
        K2_block = tl.load(
            k2_block_ptr,
            mask=mask_kv2[None, :] & mask_head_dim[:, None],
            other=0.0,
        )
        Q_block = Q_block.to(K2_block.dtype)
        K2_block = K2_block * softmax_scale
        K2_block = K2_block.to(Q_block.dtype)

        v2_block_ptr = (
            V2
            + qvk_offset
            + kv2_seq_offs[:, None] * stride_V2_seq
            + tl.arange(0, HEAD_DIM)[None, :] * stride_V2_dim
        )
        v2_block = tl.load(
            v2_block_ptr, mask=mask_kv2[:, None] & mask_head_dim[None, :], other=0.0
        )
        if convert_to_float32:
            v2_block = v2_block.to(tl.float32)

        lo_k1, hi_k1 = interval_k1(
            causal=causal,
            seq_len=SEQ_LEN,
            start_q=block_index_q * BLOCK_SIZE_Q,
            block_q=BLOCK_SIZE_Q,
            start_k2=start_kv_2,
            block_k=BLOCK_SIZE_KV,
            k1_window=k1_window,
            kk_left=kk_left,
            kk_right=kk_right,
        )

        for start_kv_1 in range(lo_k1, hi_k1, BLOCK_SIZE_KV):
            start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV)

            kv1_seq_offs = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
            mask_kv1 = kv1_seq_offs < SEQ_LEN
            mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM
            k1_block_ptr = (
                K1
                + qvk_offset
                + tl.arange(0, HEAD_DIM)[:, None] * stride_K1_dim
                + kv1_seq_offs[None, :] * stride_K1_seq
            )
            k1_block = tl.load(
                k1_block_ptr,
                mask=mask_kv1[None, :] & mask_head_dim[:, None],
                other=0.0,
            )
            k1_block = k1_block.to(Q_block.dtype)

            k1k2 = tl.reshape(
                k1_block[:, None, :] * K2_block[:, :, None],
                (HEAD_DIM_pow2, BLOCK_SIZE_KV * BLOCK_SIZE_KV),
            )
            P = tl.dot(Q_block, k1k2)

            mask_qjk = (
                mask_q[:, None, None]
                & mask_kv2[None, :, None]
                & mask_kv1[None, None, :]
            )
            if causal:
                mask_qjk = mask_qjk & get_causal_mask(
                    kk_left=kk_left,
                    kk_right=kk_right,
                    k1_window=k1_window,
                    k2_window=k2_window,
                    offs_q=offs_q,
                    offs_kv1=kv1_seq_offs,
                    offs_kv2=kv2_seq_offs,
                )
            mask_qjk = tl.reshape(
                mask_qjk, (BLOCK_SIZE_Q, BLOCK_SIZE_KV * BLOCK_SIZE_KV)
            )
            P = tl.where(mask_qjk, P, -float("inf"))
            new_m = tl.maximum(m_i, tl.max(P, axis=1))

            P_m = P - new_m[:, None]
            exp_P = tl.exp(P_m)
            l_qjk = tl.sum(exp_P, axis=1)

            v1_block_ptr = (
                V1
                + qvk_offset
                + tl.arange(0, HEAD_DIM)[None, :] * stride_V1_dim
                + kv1_seq_offs[:, None] * stride_V1_seq
            )
            v1_block = tl.load(
                v1_block_ptr,
                mask=mask_kv1[:, None] & mask_head_dim[None, :],
                other=0.0,
            )
            if convert_to_float32:
                v1_block = v1_block.to(tl.float32)
            exp_P = exp_P.to(v1_block.dtype)
            # exp_P = tl.reshape(exp_P, (BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV))

            alpha = tl.exp(m_i - new_m)
            v = tl.reshape(
                v1_block[None, :, :] * v2_block[:, None, :],
                (BLOCK_SIZE_KV * BLOCK_SIZE_KV, HEAD_DIM_pow2),
            )

            l_i = alpha * l_i + l_qjk
            m_i = new_m
            O_block = alpha[:, None] * O_block + tl.dot(exp_P, v)

    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    m_mask = offs_q < SEQ_LEN
    tl.store(m_ptrs, m_i, mask=m_mask)
    l_ptrs = L + index_batch_head * SEQ_LEN + offs_q
    l_mask = offs_q < SEQ_LEN
    tl.store(l_ptrs, l_i, mask=l_mask)

    tl.store(o_block_ptr, O_block, mask=mask_q[:, None] & mask_head_dim[None, :])
