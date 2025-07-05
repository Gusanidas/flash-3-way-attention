# @title Dq
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import triton
import triton.language as tl
import numpy as np

SHORT_CONFIG = True  # Set to True for faster development, False for production


def get_autotune_config(short=SHORT_CONFIG):
    if short:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=3, num_warps=8
            )
            for bq in [32]
            for bkv in [32]
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


@triton.autotune(configs=get_autotune_config(), key=["HEAD_DIM", "SEQ_LEN"])
@triton.jit
def _tritt_bwd_dq(
    Q,
    K1,
    K2,
    V1,
    V2,
    D,
    softmax_scale,
    dO,
    dQ,
    dK2,
    M,
    # strides for Q
    q_strideB,
    q_strideH,
    q_strideS,
    q_strideD,
    # strides for K1
    k1_strideB,
    k1_strideH,
    k1_strideS,
    k1_strideD,
    # strides for K2
    k2_strideB,
    k2_strideH,
    k2_strideS,
    k2_strideD,
    # strides for V1
    v1_strideB,
    v1_strideH,
    v1_strideS,
    v1_strideD,
    # strides for V2
    v2_strideB,
    v2_strideH,
    v2_strideS,
    v2_strideD,
    # strides for D
    d_strideB,
    d_strideH,
    d_strideS,
    # strides for dO
    dO_strideB,
    dO_strideH,
    dO_strideS,
    dO_strideD,
    # strides for dQ
    dQ_strideB,
    dQ_strideH,
    dQ_strideS,
    dQ_strideD,
    # strides for dK2
    dK2_strideB,
    dK2_strideH,
    dK2_strideS,
    dK2_strideD,
    # strides for M
    M_strideB,
    M_strideH,
    M_strideS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
    k_diff: tl.constexpr,
    convert_to_float32: tl.constexpr,
    input_precision: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    """
    Unified backward kernel to compute dQ and dK2.

    dQ = sum_{t,r} P_{q,t,r} * [(dO_q * V1_t + dO_q * V2_r) - D_q] * [K1_t * K2_r]
    dK2 gets gradients from both Q-K2 and K1-K2 attention paths
    Where:
        * P_{q,t,r} = exp(QK1K2 * softmax_scale - M_q)
        * D_q is loaded from the same shape as M => [B, N, S]
        * 'q' indexes the Q dimension, 't' indexes K1 dimension, 'r' indexes K2 dimension
    """

    block_index_q = tl.program_id(0)  # block along Q-dim
    head_idx = tl.program_id(1)  # which (batch, head)

    num_heads = NUM_HEADS
    batch_id = head_idx // num_heads
    head_id = head_idx % num_heads

    # Unified causal handling
    if CAUSAL:
        # Process both stage 1 and stage 2 ranges
        lo, hi = 0, (block_index_q + 1) * BLOCK_SIZE_Q
    else:
        # Non-causal: process full range
        lo, hi = 0, SEQ_LEN

    offset_q = batch_id * q_strideB + head_id * q_strideH
    offset_k1 = batch_id * k1_strideB + head_id * k1_strideH
    offset_k2 = batch_id * k2_strideB + head_id * k2_strideH
    offset_v1 = batch_id * v1_strideB + head_id * v1_strideH
    offset_v2 = batch_id * v2_strideB + head_id * v2_strideH
    offset_d = batch_id * d_strideB + head_id * d_strideH
    offset_dO = batch_id * dO_strideB + head_id * dO_strideH
    offset_dQ = batch_id * dQ_strideB + head_id * dQ_strideH
    offset_dK2 = batch_id * dK2_strideB + head_id * dK2_strideH
    offset_m = batch_id * M_strideB + head_id * M_strideH

    start_q = block_index_q * BLOCK_SIZE_Q
    offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)
    offs_dim = tl.arange(0, HEAD_DIM)

    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(
        Q + (offset_q + offs_q[:, None] * q_strideS + offs_dim[None, :] * q_strideD),
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
        other=0.0,
    )
    if convert_to_float32:
        Q_block = Q_block.to(tl.float32)

    # Load dO block => shape [BLOCK_Q, HEAD_DIM]
    dO_block = tl.load(
        dO
        + (offset_dO + offs_q[:, None] * dO_strideS + offs_dim[None, :] * dO_strideD),
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
        other=0.0,
    )
    if convert_to_float32:
        dO_block = dO_block.to(tl.float32)

    # D => shape [BLOCK_Q]; D is [B, N, S]
    D_block = tl.load(
        D + (offset_d + offs_q * d_strideS), mask=offs_q < SEQ_LEN, other=0.0
    )
    if convert_to_float32:
        D_block = D_block.to(tl.float32)

    # M => shape [BLOCK_Q]; M is [B, N, S]
    m_q = tl.load(M + (offset_m + offs_q * M_strideS), mask=offs_q < SEQ_LEN, other=0.0)
    if convert_to_float32:
        m_q = m_q.to(tl.float32)
    avg_m = tl.max(m_q)

    # Loop over K1 in blocks
    for start_kv_2 in range(lo, hi, BLOCK_SIZE_KV):
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV)

        lo_1_k_diff = BLOCK_SIZE_KV * (
            tl.maximum(0, start_kv_2 - k_diff) // BLOCK_SIZE_KV
        )
        if CAUSAL:
            k1_hi = start_kv_2 + BLOCK_SIZE_KV
            lo_1 = tl.maximum(lo_1_k_diff, 0)
        else:
            k1_hi = hi
            lo_1 = 0

        current_offs_k2_seq = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        current_k2_block_ptr = (
            K2
            + offset_k2
            + tl.arange(0, HEAD_DIM)[:, None] * k2_strideD
            + current_offs_k2_seq[None, :] * k2_strideS
        )
        mask_k2 = current_offs_k2_seq < SEQ_LEN
        K2_block = tl.load(current_k2_block_ptr, mask=mask_k2[None, :], other=0.0)
        if convert_to_float32:
            K2_block = K2_block.to(tl.float32)

        acc_k2v1 = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
        acc_pk2 = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32)

        current_offs_v2_seq = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        current_v2_block_ptr = (
            V2
            + offset_v2
            + current_offs_v2_seq[None, :] * v2_strideS
            + tl.arange(0, HEAD_DIM)[:, None] * v2_strideD
        )
        mask_v2 = current_offs_v2_seq < SEQ_LEN
        V2_block = tl.load(current_v2_block_ptr, mask=mask_v2[None, :], other=0.0)
        if convert_to_float32:
            V2_block = V2_block.to(tl.float32)

        for start_kv_1 in range(lo_1, k1_hi, BLOCK_SIZE_KV):
            start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV)

            current_offs_k1_seq = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
            current_k1_block_ptr = (
                K1
                + offset_k1
                + current_offs_k1_seq[:, None] * k1_strideS
                + tl.arange(0, HEAD_DIM)[None, :] * k1_strideD
            )
            mask_k1 = current_offs_k1_seq < SEQ_LEN
            K1_block = tl.load(current_k1_block_ptr, mask=mask_k1[:, None], other=0.0)
            if convert_to_float32:
                K1_block = K1_block.to(tl.float32)

            current_offs_v1_seq = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
            current_v1_block_ptr = (
                V1
                + offset_v1
                + current_offs_v1_seq[None, :] * v1_strideS
                + tl.arange(0, HEAD_DIM)[:, None] * v1_strideD
            )
            mask_v1 = current_offs_v1_seq < SEQ_LEN
            V1_block = tl.load(current_v1_block_ptr, mask=mask_v1[None, :], other=0.0)
            if convert_to_float32:
                V1_block = V1_block.to(tl.float32)
            if input_precision is not None:
                K1K2_block = tl.dot(
                    K1_block,
                    K2_block.to(K1_block.dtype),
                    input_precision=input_precision,
                )
            else:
                K1K2_block = tl.dot(K1_block, K2_block.to(K1_block.dtype))

            mask_k1_k2 = mask_k1[:, None] & mask_k2[None, :]
            if CAUSAL:
                mask_k1_lt_k2 = (
                    current_offs_k1_seq[:, None] <= current_offs_k2_seq[None, :]
                )  # [BLOCK_SIZE_KV, BLOCK_SIZE_KV]
                mask_k1_k2_k_diff = (
                    current_offs_k1_seq[:, None] + k_diff
                    >= current_offs_k2_seq[None, :]
                )  # [BLOCK_SIZE_KV, BLOCK_SIZE_KV]
                mask_k1_lt_k2 = mask_k1_lt_k2 & mask_k1_k2_k_diff
                mask_k1_k2 = mask_k1_k2 & mask_k1_lt_k2

            K1K2_block = K1K2_block * softmax_scale - 0.5 * avg_m

            K1K2_block = tl.where(mask_k1_k2, K1K2_block, float("-inf"))
            Pk_block = tl.math.exp(K1K2_block)
            Pk_block = Pk_block * tl.where(mask_k1_k2, 1, 0)
            Pk_block = Pk_block.to(dO_block.dtype)
            if input_precision is not None:
                acc_k2v1 += tl.trans(
                    tl.dot(V1_block, Pk_block, input_precision=input_precision)
                )
            else:
                acc_k2v1 += tl.trans(tl.dot(V1_block, Pk_block))
            acc_pk2 += tl.sum(Pk_block, 0)

        if input_precision is not None:
            QK2_block = tl.dot(Q_block, K2_block, input_precision=input_precision)
        else:
            QK2_block = tl.dot(Q_block, K2_block)
        mask_q_k2 = (offs_q[:, None] < SEQ_LEN) & mask_k2[None, :]
        if CAUSAL:
            mask_q_gt_k2 = offs_q[:, None] >= current_offs_k2_seq[None, :]
            mask_q_k2 = mask_q_k2 & mask_q_gt_k2

        QK2_block = QK2_block * softmax_scale - m_q[:, None]
        QK2_block = tl.where(mask_q_k2, QK2_block, float("-inf"))
        P_qk = tl.math.exp(QK2_block)
        P_qk = P_qk * tl.where(mask_q_k2, 1, 0)
        P_v1v2 = acc_k2v1 * tl.trans(V2_block)
        if not convert_to_float32:
            P_v1v2 = P_v1v2.to(Q_block.dtype)
            P_qk = P_qk.to(Q_block.dtype)
        if input_precision is not None:
            P_tot = P_qk * tl.dot(
                dO_block, tl.trans(P_v1v2), input_precision=input_precision
            )
        else:
            P_tot = P_qk * tl.dot(dO_block, tl.trans(P_v1v2))
        P_d = P_qk * D_block[:, None] * acc_pk2[None, :]

        P_qk_ds = P_tot - P_d
        P_qk_ds = P_qk_ds * tl.exp(0.5 * (avg_m))
        if not convert_to_float32:
            P_qk_ds = P_qk_ds.to(K2_block.dtype)
        if input_precision is not None:
            dQ_block += softmax_scale * tl.dot(
                P_qk_ds, tl.trans(K2_block), input_precision=input_precision
            )
        else:
            dQ_block += softmax_scale * tl.dot(P_qk_ds, tl.trans(K2_block))

        # Accumulate dK2 gradients from Q-K2 path
        if input_precision is not None:
            acc_dK2_from_qk = softmax_scale * tl.dot(
                tl.trans(P_qk_ds), Q_block, input_precision=input_precision
            )
        else:
            acc_dK2_from_qk = softmax_scale * tl.dot(tl.trans(P_qk_ds), Q_block)

        # Write dK2 gradients from Q-K2 path using atomic_add
        current_dK2_block_ptr = (
            dK2
            + offset_dK2
            + current_offs_k2_seq[:, None] * dK2_strideS
            + tl.arange(0, HEAD_DIM)[None, :] * dK2_strideD
        )
        mask_k2_dim = mask_k2[:, None]  # & (offs_dim[None, :] < HEAD_DIM)
        tl.atomic_add(
            current_dK2_block_ptr,
            acc_dK2_from_qk,
            mask=mask_k2_dim,
        )

    # Store the dQ result
    tl.store(
        dQ
        + (offset_dQ + offs_q[:, None] * dQ_strideS + offs_dim[None, :] * dQ_strideD),
        dQ_block,
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
    )


def tritt_bwd_dq(
    Q,
    K1,
    K2,
    V1,
    V2,
    D,
    softmax_scale,
    dO,
    M,
    causal,
    k_diff=2048,
    convert_to_float32=False,
    input_precision=None,
):
    """
    Wrapper function for backward dQ and dK2 computation.

    Args:
        Q, K1, K2, V1, V2: Input tensors
        D: Preprocessing result from forward pass
        softmax_scale: Scaling factor for attention
        dO: Gradient of output
        M: Logits normalization from forward pass
        causal: Whether to use causal masking
        k_diff: Maximum distance for k-masking

    Returns:
        dQ: Gradient w.r.t. Q
        dK2: Gradient w.r.t. K2
    """
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    dQ = torch.zeros_like(Q)
    dK2 = torch.zeros_like(K2)
    V1 = F.silu(V1)
    V1 = V1.contiguous()
    V2 = V2.contiguous()
    Q = Q.contiguous()
    K1 = K1.contiguous()
    K2 = K2.contiguous()
    D = D.contiguous()
    M = M.contiguous()
    dO = dO.contiguous()
    dQ = dQ.contiguous()
    dK2 = dK2.contiguous()

    # Grid configuration

    # Block sizes
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_KV = 32
    grid = lambda META: (triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q), BATCH_SIZE * NUM_HEADS)

    # Compute strides
    q_strides = Q.stride()
    k1_strides = K1.stride()
    k2_strides = K2.stride()
    v1_strides = V1.stride()
    v2_strides = V2.stride()
    do_strides = dO.stride()
    d_strides = D.stride()
    m_strides = M.stride()
    dq_strides = dQ.stride()
    dk2_strides = dK2.stride()

    # Call the unified kernel
    _tritt_bwd_dq[grid](
        Q,
        K1,
        K2,
        V1,
        V2,
        D,
        softmax_scale,
        dO,
        dQ,
        dK2,
        M,
        # Q strides
        q_strides[0],
        q_strides[1],
        q_strides[2],
        q_strides[3],
        # K1 strides
        k1_strides[0],
        k1_strides[1],
        k1_strides[2],
        k1_strides[3],
        # K2 strides
        k2_strides[0],
        k2_strides[1],
        k2_strides[2],
        k2_strides[3],
        # V1 strides
        v1_strides[0],
        v1_strides[1],
        v1_strides[2],
        v1_strides[3],
        # V2 strides
        v2_strides[0],
        v2_strides[1],
        v2_strides[2],
        v2_strides[3],
        # D strides
        d_strides[0],
        d_strides[1],
        d_strides[2],
        # dO strides
        do_strides[0],
        do_strides[1],
        do_strides[2],
        do_strides[3],
        # dQ strides
        dq_strides[0],
        dq_strides[1],
        dq_strides[2],
        dq_strides[3],
        # dK2 strides
        dk2_strides[0],
        dk2_strides[1],
        dk2_strides[2],
        dk2_strides[3],
        # M strides
        m_strides[0],
        m_strides[1],
        m_strides[2],
        SEQ_LEN,
        HEAD_DIM,
        NUM_HEADS,
        causal,
        k_diff,
        convert_to_float32,
        input_precision,
    )

    return dQ, dK2


if __name__ == "__main__":
    # Test the backward dQ kernel with small inputs
    torch.manual_seed(42)

    # Small test dimensions
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = 1, 2, 30, 32
    causal = False
    device = "cuda"
    dtype = torch.bfloat16

    # Create random test inputs
    Q = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )
    K1 = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )
    K2 = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )
    V1 = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )
    V2 = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )

    # D and M should have shape [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    D = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, dtype=dtype, device=device)
    M = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, dtype=dtype, device=device)

    # dO (gradient of output) has same shape as Q
    dO = torch.randn(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device
    )

    softmax_scale = 1.0 / (HEAD_DIM**0.5)
    softmax_scale = 1.0

    print("Testing tritt_bwd_dq function...")
    print(f"Input shapes: Q={Q.shape}, K1={K1.shape}, K2={K2.shape}")
    print(f"V1={V1.shape}, V2={V2.shape}, D={D.shape}, M={M.shape}, dO={dO.shape}")

    # Test non-causal case
    try:
        dQ, dK2 = tritt_bwd_dq(Q, K1, K2, V1, V2, D, softmax_scale, dO, M, causal=False)
        print(f"Non-causal dQ shape: {dQ.shape}, dK2 shape: {dK2.shape}")
        print(f"Non-causal dQ mean: {dQ.mean().item():.6f}, std: {dQ.std().item():.6f}")
        print(
            f"Non-causal dK2 mean: {dK2.mean().item():.6f}, std: {dK2.std().item():.6f}"
        )
    except Exception as e:
        print(f"Non-causal test failed: {e}")

    # Test causal case
    dQ_causal, dK2_causal = tritt_bwd_dq(
        Q, K1, K2, V1, V2, D, softmax_scale, dO, M, causal=True
    )
    print(f"Causal dQ shape: {dQ_causal.shape}, dK2 shape: {dK2_causal.shape}")
    print(
        f"Causal dQ mean: {dQ_causal.mean().item():.6f}, std: {dQ_causal.std().item():.6f}"
    )
    print(
        f"Causal dK2 mean: {dK2_causal.mean().item():.6f}, std: {dK2_causal.std().item():.6f}"
    )

    print("Test completed!")
