import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from .utils import process_masking_variables
from .intervals import interval_k1, interval_k2_from_q

SHORT_CONFIG = True  # Set to True for faster development, False for production


def get_autotune_config(short=SHORT_CONFIG):
    if short:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=3, num_warps=8
            )
            for bq in [16]
            for bkv in [16]
        ]
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw
            )
            for bq in [16, 32, 64, 128]
            for bkv in [16, 32, 64, 128]
            for ns in [1, 2, 3]
            for nw in [1, 2, 4, 8]
        ]


@triton.jit
def _get_mask(
    kk_left,
    kk_right,
    k1_window,
    k2_window,
    offs_q,
    offs_kv1,
    offs_kv2,
):
    qk1_mask = (offs_q[:, None] <= offs_kv1[None, :] + k1_window) & (
        offs_q[:, None] >= offs_kv1[None, :]
    )
    qk2_mask = (offs_q[:, None] <= offs_kv2[None, :] + k2_window) & (
        offs_q[:, None] >= offs_kv2[None, :]
    )
    kk_mask = (offs_kv1[None, :] + kk_left >= offs_kv2[:, None]) & (
        offs_kv2[:, None] + kk_right >= offs_kv1[None, :]
    )
    return qk1_mask[:, None, :] & qk2_mask[:, :, None] & kk_mask[None, :, :]


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
                mask_qjk = mask_qjk & _get_mask(
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
            new_m = new_m

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


def tritt_fwd(
    Q,
    K1,
    K2,
    V1,
    V2,
    softmax_scale,
    kk_left,
    kk_right,
    k1_window,
    k2_window,
    causal,
    convert_to_float32,
):
    (
        BATCH_SIZE,
        NUM_HEADS,
        SEQ_LEN,
        HEAD_DIM,
    ) = Q.shape
    HEAD_DIM_pow2 = triton.next_power_of_2(HEAD_DIM)
    O = torch.zeros(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=Q.dtype,
        device=Q.device,
    )
    M = torch.zeros(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN),
        dtype=torch.float32,
        device=Q.device,
    )
    L = torch.zeros(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN),
        dtype=torch.float32,
        device=Q.device,
    )
    grid = lambda meta: (
        triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
    )
    V1 = F.silu(V1)

    Q = Q.contiguous()
    K1 = K1.contiguous()
    K2 = K2.contiguous()
    V1 = V1.contiguous()
    V2 = V2.contiguous()
    O = O.contiguous()
    M = M.contiguous()
    L = L.contiguous()

    k1_window, k2_window, kk_left, kk_right = process_masking_variables(
        SEQ_LEN, k1_window, k2_window, kk_left, kk_right
    )

    sqrt_softmax_scale = softmax_scale**0.5

    _tritt_fwd[grid](
        Q=Q,
        K1=K1,
        K2=K2,
        V1=V1,
        V2=V2,
        softmax_scale=sqrt_softmax_scale,
        M=M,
        L=L,
        O=O,
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
        stride_O_seq=O.stride(2),
        stride_O_dim=O.stride(3),
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
        HEAD_DIM_pow2=HEAD_DIM_pow2,
        kk_left=kk_left,
        kk_right=kk_right,
        k1_window=k1_window,
        k2_window=k2_window,
        causal=causal,
        convert_to_float32=convert_to_float32,
    )
    return O, M, L
