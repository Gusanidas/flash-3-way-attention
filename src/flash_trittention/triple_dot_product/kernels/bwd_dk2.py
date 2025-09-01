import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from .utils import process_masking_variables
from .intervals import interval_k1_from_k2, intervals_q
from .utils import (
    get_autotune_config,
    get_causal_mask,
)


@triton.autotune(configs=get_autotune_config(short=True), key=["HEAD_DIM", "SEQ_LEN"])
@triton.jit
def _tritt_bwd_dk2(
    Q,
    K1,
    K2,
    V1,
    V2,
    dO,
    D,
    M,
    dk2,
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
    stride_dk2_seq,
    stride_dk2_dim,
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
    block_index_k2 = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    md_offset = index_batch_head * SEQ_LEN
    M = M + md_offset
    D = D + md_offset

    start_kv_2 = block_index_k2 * BLOCK_SIZE_KV

    kv2_seq_offs = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
    mask_kv2 = kv2_seq_offs < SEQ_LEN
    mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM

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
    # May cause problems later converting to float32
    softmax_scale = softmax_scale.to(K2_block.dtype)
    if convert_to_float32:
        K2_block = K2_block.to(tl.float32)

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

    dk2_block = tl.zeros([HEAD_DIM_pow2, BLOCK_SIZE_KV], dtype=tl.float32)

    lo_k1, hi_k1 = interval_k1_from_k2(
        causal=causal,
        seq_len=SEQ_LEN,
        start_k2=start_kv_2,
        block_k=BLOCK_SIZE_KV,
        k2_window=k2_window,
        kk_left=kk_left,
        kk_right=kk_right,
    )

    for start_kv_1 in range(lo_k1, hi_k1, BLOCK_SIZE_KV):
        start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV)

        kv1_seq_offs = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
        mask_kv1 = kv1_seq_offs < SEQ_LEN

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
        if convert_to_float32:
            k1_block = k1_block.to(tl.float32)

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

        k1k2 = tl.reshape(
            k1_block[:, None, :] * K2_block[:, :, None],
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
            start_q = tl.multiple_of(start_q, BLOCK_SIZE_Q)

            offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)
            mask_q = offs_q < SEQ_LEN

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

            offset = offs_q
            m_mask = offs_q < SEQ_LEN
            m_i = tl.load(M + offset, mask=m_mask, other=0.0)
            D_i = tl.load(D + offset, mask=m_mask, other=0.0)

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
            P = P.to(dO_block.dtype)

            # Apply masking
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
            P = tl.where(mask_qjk, P, -masking_value)

            P = tl.exp(P - m_i[:, None])
            P = P.to(v1_block.dtype)
            P = tl.reshape(P, (BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV))

            # Compute V1*V2 for dP computation
            v = tl.trans(
                tl.reshape(
                    v1_block[None, :, :] * v2_block[:, None, :],
                    (BLOCK_SIZE_KV * BLOCK_SIZE_KV, HEAD_DIM_pow2),
                )
            )

            # Compute dP = dO*V1*V2
            dP = tl.dot(dO_block, v)
            dP = dP.to(D_i.dtype)
            dP = tl.reshape(dP, (BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_KV))

            # Compute dS = P * (dP - D)
            dS = P * (dP - D_i[:, None, None])

            # Compute dk2 gradient: dk2 += QK1^T * dS * softmax_scale
            dS_reshaped = tl.reshape(
                tl.trans(dS, (0, 2, 1)), (BLOCK_SIZE_Q * BLOCK_SIZE_KV, BLOCK_SIZE_KV)
            )
            QK1 = tl.reshape(
                tl.trans(Q_block)[:, :, None] * k1_block[:, None, :],
                (HEAD_DIM_pow2, BLOCK_SIZE_Q * BLOCK_SIZE_KV),
            )
            dk2_block += tl.dot(QK1, dS_reshaped)

    kv2_seq_offs = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
    mask_kv2 = kv2_seq_offs < SEQ_LEN
    mask_head_dim = tl.arange(0, HEAD_DIM_pow2) < HEAD_DIM

    dk2_block_ptr = (
        dk2
        + qvk_offset
        + kv2_seq_offs[None, :] * stride_dk2_seq
        + tl.arange(0, HEAD_DIM)[:, None] * stride_dk2_dim
    )
    tl.store(dk2_block_ptr, dk2_block, mask=mask_kv2[None, :] & mask_head_dim[:, None])


def tritt_bwd_dk2(
    Q,
    K1,
    K2,
    V1,
    V2,
    dO,
    O,
    M,
    causal,
    softmax_scale,
    seq_len,
    batch_size,
    num_heads,
    head_dim,
    k1_window,
    k2_window,
    kk_left,
    kk_right,
):
    Q = Q.contiguous()
    K1 = K1.contiguous()
    K2 = K2.contiguous()
    V1 = V1.contiguous()
    V2 = V2.contiguous()
    dO = dO.contiguous()
    O = O.contiguous()
    M = M.contiguous()

    V1 = F.silu(V1)

    dk2 = torch.zeros_like(K2)

    D = torch.sum(dO * O, dim=-1)

    HEAD_DIM_pow2 = triton.next_power_of_2(head_dim)

    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_SIZE_KV"]),
        batch_size * num_heads,
    )

    k1_window, k2_window, kk_left, kk_right = process_masking_variables(
        seq_len, k1_window, k2_window, kk_left, kk_right
    )

    convert_to_float32 = True
    masking_value = 1e-4 if Q.dtype == torch.float32 else 1e-2

    _tritt_bwd_dk2[grid](
        Q=Q,
        K1=K1,
        K2=K2,
        V1=V1,
        V2=V2,
        dO=dO,
        D=D,
        M=M,
        dk2=dk2,
        softmax_scale=softmax_scale**0.5,
        stride_Q_batch=Q.stride(0),
        stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),
        stride_Q_dim=Q.stride(3),
        stride_K1_seq=K1.stride(2),
        stride_K1_dim=K1.stride(3),
        stride_K2_seq=K2.stride(2),
        stride_K2_dim=K2.stride(3),
        stride_V1_seq=V1.stride(2),
        stride_V1_dim=V1.stride(3),
        stride_V2_seq=V2.stride(2),
        stride_V2_dim=V2.stride(3),
        stride_dO_seq=dO.stride(2),
        stride_dO_dim=dO.stride(3),
        stride_dk2_seq=dk2.stride(2),
        stride_dk2_dim=dk2.stride(3),
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        HEAD_DIM_pow2=HEAD_DIM_pow2,
        kk_left=kk_left,
        kk_right=kk_right,
        k1_window=k1_window,
        k2_window=k2_window,
        causal=causal,
        convert_to_float32=convert_to_float32,
        masking_value=masking_value,
    )
    return dk2
