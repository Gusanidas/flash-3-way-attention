import triton
import triton.language as tl
from .intervals import interval_k2_from_k1, intervals_q
from .utils import get_autotune_config, get_causal_mask


@triton.autotune(configs=get_autotune_config(short=True), key=["HEAD_DIM", "SEQ_LEN"])
@triton.jit
def _tritt_bwd_dv1(
    Q,
    K1,
    K2,
    V1,
    V2,
    dO,
    M,
    dv1,
    softmax_scale,
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
    stride_dO_seq,
    stride_dO_dim,
    stride_dv1_seq,
    stride_dv1_dim,
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
    masking_value: tl.constexpr,
):
    block_index_k1 = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    md_offset = index_batch_head * SEQ_LEN
    M = M + md_offset

    start_kv_1 = block_index_k1 * BLOCK_SIZE_KV

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
    # May cause problems later converting to float32
    softmax_scale = softmax_scale.to(k1_block.dtype)
    if convert_to_float32:
        k1_block = k1_block.to(tl.float32)

    dv1_block = tl.zeros([HEAD_DIM_pow2, BLOCK_SIZE_KV], dtype=tl.float32)

    lo_k2, hi_k2 = interval_k2_from_k1(
        causal=causal,
        seq_len=SEQ_LEN,
        start_k1=start_kv_1,
        block_k=BLOCK_SIZE_KV,
        k1_window=k1_window,
        kk_left=kk_left,
        kk_right=kk_right,
    )

    for start_kv_2 in range(lo_k2, hi_k2, BLOCK_SIZE_KV):
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV)

        kv2_seq_offs = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        mask_kv2 = kv2_seq_offs < SEQ_LEN

        k2_block_ptr = (
            K2
            + qvk_offset
            + tl.arange(0, HEAD_DIM)[:, None] * stride_K2_dim
            + kv2_seq_offs[None, :] * stride_K2_seq
        )
        k2_block = tl.load(
            k2_block_ptr,
            mask=mask_kv2[None, :] & mask_head_dim[:, None],
            other=0.0,
        )
        if convert_to_float32:
            k2_block = k2_block.to(tl.float32)

        # Load V2 block
        v2_block_ptr = (
            V2
            + qvk_offset
            + tl.arange(0, HEAD_DIM)[None, :] * stride_V2_dim
            + kv2_seq_offs[:, None] * stride_V2_seq
        )
        v2_block = tl.load(
            v2_block_ptr,
            mask=mask_kv2[:, None] & mask_head_dim[None, :],
            other=0.0,
        )
        if convert_to_float32:
            v2_block = v2_block.to(tl.float32)

        # Compute K1*K2
        k1k2 = tl.reshape(
            k1_block[:, :, None] * k2_block[:, None, :],
            (HEAD_DIM_pow2, BLOCK_SIZE_KV * BLOCK_SIZE_KV),
        )
        lo_q, hi_q = intervals_q(
            causal=causal,
            seq_len=SEQ_LEN,
            start_k1=start_kv_1,
            block_k=BLOCK_SIZE_KV,
            start_k2=start_kv_2,
            k1_window=k1_window,
            k2_window=k2_window,
            kk_left=kk_left,
            kk_right=kk_right,
        )

        for start_q in range(lo_q, hi_q, BLOCK_SIZE_Q):
            offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)
            mask_q = offs_q < SEQ_LEN

            # Load Q block
            q_block_ptr = (
                Q
                + qvk_offset
                + offs_q[:, None] * stride_Q_seq
                + tl.arange(0, HEAD_DIM_pow2)[None, :] * stride_Q_dim
            )
            Q_block = tl.load(
                q_block_ptr, mask=mask_q[:, None] & mask_head_dim[None, :], other=0.0
            )
            Q_block = Q_block * softmax_scale
            if convert_to_float32:
                Q_block = Q_block.to(tl.float32)

            # Load M values for this block
            offset = offs_q
            m_mask = offs_q < SEQ_LEN
            m_i = tl.load(M + offset, mask=m_mask, other=float("inf"))

            # Load dO block
            dO_block_ptr = (
                dO
                + qvk_offset
                + offs_q[:, None] * stride_dO_seq
                + tl.arange(0, HEAD_DIM_pow2)[None, :] * stride_dO_dim
            )
            dO_block = tl.load(
                dO_block_ptr, mask=mask_q[:, None] & mask_head_dim[None, :], other=0.0
            )
            if convert_to_float32:
                dO_block = dO_block.to(tl.float32)

            # Compute P = Q*K1*K2
            P = tl.dot(Q_block, k1k2)

            # Apply masking. Build the sequence-bounds mask in the same
            # (Q, kv2, kv1) layout as get_causal_mask so the shared transpose
            # below moves both parts to (Q, kv1, kv2) consistently.
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
            mask_qjk = tl.trans(mask_qjk, (0, 2, 1))
            mask_qjk = tl.reshape(
                mask_qjk, (BLOCK_SIZE_Q, BLOCK_SIZE_KV * BLOCK_SIZE_KV)
            )
            P = tl.where(mask_qjk, P, -masking_value)

            # Apply softmax with stored m values
            P = tl.exp(P - m_i[:, None])
            P = P.to(dO_block.dtype)
            P = tl.reshape(P, (BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV))

            ## Compute dV = tl.dot(P, dO) in the q dimension
            P_reshaped = tl.reshape(
                tl.trans(P, (0, 2, 1)), (BLOCK_SIZE_Q * BLOCK_SIZE_KV, BLOCK_SIZE_KV)
            )
            dOK2 = tl.reshape(
                tl.trans(dO_block)[:, :, None] * tl.trans(v2_block)[:, None, :],
                (HEAD_DIM_pow2, BLOCK_SIZE_Q * BLOCK_SIZE_KV),
            )
            dv1_block += tl.dot(dOK2, P_reshaped)

    kv1_seq_offs = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
    mask_kv1 = kv1_seq_offs < SEQ_LEN
    mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM

    dv1_block_ptr = (
        dv1
        + qvk_offset
        + kv1_seq_offs[:, None] * stride_dv1_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_dv1_dim
    )
    tl.store(
        dv1_block_ptr,
        tl.trans(dv1_block),
        mask=mask_kv1[:, None] & mask_head_dim[None, :],
    )
