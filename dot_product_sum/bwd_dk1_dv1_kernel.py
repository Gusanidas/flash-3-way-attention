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
            for bq in [16]
            for bkv in [32]
        ]
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw
            )
            for bq in [16, 32, 64]
            for bkv in [16, 32, 64]
            for ns in [2, 3]
            for nw in [2, 4, 8]
        ]


@triton.autotune(configs=get_autotune_config(), key=["HEAD_DIM", "SEQ_LEN"])
@triton.jit
def _tritt_bwd_dk1_dv1(
    Q,
    K1,
    K2,
    V1,
    V2,
    D,
    softmax_scale,
    dO,
    M,
    dK1,
    dV1,
    dV2,
    dK2,
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
    # strides for M
    M_strideB,
    M_strideH,
    M_strideS,
    # strides for dK1
    dK1_strideB,
    dK1_strideH,
    dK1_strideS,
    dK1_strideD,
    # strides for dV1
    dV1_strideB,
    dV1_strideH,
    dV1_strideS,
    dV1_strideD,
    # strides for dV2
    dV2_strideB,
    dV2_strideH,
    dV2_strideS,
    dV2_strideD,
    # strides for dK2
    dK2_strideB,
    dK2_strideH,
    dK2_strideS,
    dK2_strideD,
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
    Kernel for computing dK1, dV1, dV2, and dK2.
    """
    # Get program IDs (moved from outer kernel)
    batch_head_id = tl.program_id(0)  # id for batch and head
    k1_v1_block_id = tl.program_id(1)  # id for k1 and v1 (same one)

    # Extract batch and head indices
    batch_id = batch_head_id // NUM_HEADS
    head_id = batch_head_id % NUM_HEADS

    # Calculate k1 start position
    k1_start = k1_v1_block_id * BLOCK_SIZE_KV
    # Calculate base offsets
    offset_q = batch_id * q_strideB + head_id * q_strideH
    offset_k1 = batch_id * k1_strideB + head_id * k1_strideH
    offset_k2 = batch_id * k2_strideB + head_id * k2_strideH
    offset_v1 = batch_id * v1_strideB + head_id * v1_strideH
    offset_v2 = batch_id * v2_strideB + head_id * v2_strideH
    offset_d = batch_id * d_strideB + head_id * d_strideH
    offset_dO = batch_id * dO_strideB + head_id * dO_strideH
    offset_m = batch_id * M_strideB + head_id * M_strideH
    offset_dK1 = batch_id * dK1_strideB + head_id * dK1_strideH
    offset_dV1 = batch_id * dV1_strideB + head_id * dV1_strideH
    offset_dV2 = batch_id * dV2_strideB + head_id * dV2_strideH

    # Load K1 and V1 blocks
    offs_k1 = k1_start + tl.arange(0, BLOCK_SIZE_KV)
    offs_dim = tl.arange(0, HEAD_DIM)
    mask_k1 = offs_k1 < SEQ_LEN
    K1_block = tl.load(
        K1 + offset_k1 + offs_k1[:, None] * k1_strideS + offs_dim[None, :] * k1_strideD,
        mask=mask_k1[:, None] & (offs_dim[None, :] < HEAD_DIM),
        other=0.0,
    )
    if convert_to_float32:
        K1_block = K1_block.to(tl.float32)

    V1_block = tl.load(
        V1 + offset_v1 + offs_k1[None, :] * v1_strideS + offs_dim[:, None] * v1_strideD,
        mask=mask_k1[None, :] & (offs_dim[:, None] < HEAD_DIM),
        other=0.0,
    )
    if convert_to_float32:
        V1_block = V1_block.to(tl.float32)

    # Initialize accumulators for dK1 and dV1
    dK1_acc = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dV1_acc = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    # Define looping limits for k2
    if CAUSAL:
        k2_lo = k1_start
        k2_hi = tl.minimum(k1_start + k_diff + BLOCK_SIZE_KV + 1, SEQ_LEN)
    else:
        k2_lo = 0
        k2_hi = SEQ_LEN

    # Loop over k2
    for k2_start_loop in range(k2_lo, k2_hi, BLOCK_SIZE_KV):
        k2_start_loop = tl.multiple_of(k2_start_loop, BLOCK_SIZE_KV)
        offs_k2 = k2_start_loop + tl.arange(0, BLOCK_SIZE_KV)
        mask_k2 = offs_k2 < SEQ_LEN

        # Load K2 and V2
        K2_block = tl.load(
            K2
            + offset_k2
            + offs_k2[:, None] * k2_strideS
            + offs_dim[None, :] * k2_strideD,
            mask=mask_k2[:, None] & (offs_dim[None, :] < HEAD_DIM),
            other=0.0,
        )
        if convert_to_float32:
            K2_block = K2_block.to(tl.float32)

        V2_block = tl.load(
            V2
            + offset_v2
            + offs_k2[None, :] * v2_strideS
            + offs_dim[:, None] * v2_strideD,
            mask=mask_k2[None, :] & (offs_dim[:, None] < HEAD_DIM),
            other=0.0,
        )
        if convert_to_float32:
            V2_block = V2_block.to(tl.float32)

        acc_QK2O = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
        # For the term that will be subtracted.
        acc_QDK2_sub = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32)
        acc_dOv1 = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

        # Define looping limits for q
        if CAUSAL:
            q_lo = k2_start_loop
            q_hi = SEQ_LEN
        else:
            q_lo = 0
            q_hi = SEQ_LEN

        # Loop over q
        for q_start in range(q_lo, q_hi, BLOCK_SIZE_Q):
            q_start = tl.multiple_of(q_start, BLOCK_SIZE_Q)
            offs_q = q_start + tl.arange(0, BLOCK_SIZE_Q)
            mask_q = offs_q < SEQ_LEN

            # Load D, M, Q and dO
            D_block = tl.load(
                D + offset_d + offs_q * d_strideS,
                mask=mask_q,
                other=0.0,
            )
            if convert_to_float32:
                D_block = D_block.to(tl.float32)

            M_block = tl.load(
                M + offset_m + offs_q * M_strideS,
                mask=mask_q,
                other=0.0,
            )
            if convert_to_float32:
                M_block = M_block.to(tl.float32)

            Q_block = tl.load(
                Q
                + offset_q
                + offs_q[:, None] * q_strideS
                + offs_dim[None, :] * q_strideD,
                mask=mask_q[:, None] & (offs_dim[None, :] < HEAD_DIM),
                other=0.0,
            )
            if convert_to_float32:
                Q_block = Q_block.to(tl.float32)

            dO_block = tl.load(
                dO
                + offset_dO
                + offs_q[:, None] * dO_strideS
                + offs_dim[None, :] * dO_strideD,
                mask=mask_q[:, None] & (offs_dim[None, :] < HEAD_DIM),
                other=0.0,
            )
            if convert_to_float32:
                dO_block = dO_block.to(tl.float32)

            # QK2 = tl.dot(Q, K2) * softmax_scale - M
            if input_precision is not None:
                QK2_block = (
                    tl.dot(Q_block, tl.trans(K2_block), input_precision=input_precision)
                    * softmax_scale
                    - M_block[:, None]
                )
            else:
                QK2_block = (
                    tl.dot(Q_block, tl.trans(K2_block)) * softmax_scale
                    - M_block[:, None]
                )

            # Do necessary masking if causal
            maskqk = mask_q[:, None] & mask_k2[None, :]
            if CAUSAL:
                mask_causal = offs_q[:, None] >= offs_k2[None, :]
                maskqk = maskqk & mask_causal
            QK2_block = tl.where(maskqk, QK2_block, float("-inf"))

            # PQK2 = tl.exp(QK2)
            PQK2_block = tl.exp(QK2_block)

            if not convert_to_float32:
                PQK2_block = PQK2_block.to(Q_block.dtype)

            if input_precision is not None:
                acc_QK2O += tl.dot(
                    tl.trans(PQK2_block), dO_block, input_precision=input_precision
                )
            else:
                acc_QK2O += tl.dot(tl.trans(PQK2_block), dO_block)
            acc_QDK2_sub += tl.sum(PQK2_block * D_block[:, None], 0)

            # dv += PQK2 * dO and sum along the q dimension
            # acc_dv += tl.sum(
            #    PQK2_block[:, :, None] * dO_block[:, None, :], 0
            # )  # tl dot?
            acc_dOv1 += tl.dot(tl.trans(PQK2_block), dO_block)

        # K1K2 = tl.dot(K1, K2) * softmax_scale
        if input_precision is not None:
            K1K2_block = (
                tl.dot(K1_block, tl.trans(K2_block), input_precision=input_precision)
                * softmax_scale
            )
        else:
            K1K2_block = tl.dot(K1_block, tl.trans(K2_block)) * softmax_scale

        # Do necessary masking if causal
        maskk1k2 = mask_k1[:, None] & mask_k2[None, :]
        if CAUSAL:
            mask_causal = offs_k1[:, None] <= offs_k2[None, :]
            mask_k1_k2_k_diff = offs_k1[:, None] + k_diff >= offs_k2[None, :]
            maskk1k2 = (maskk1k2 & mask_causal) & mask_k1_k2_k_diff
        K1K2_block = tl.where(maskk1k2, K1K2_block, float("-inf"))

        # PK1K2 = tl.exp(K1K2)
        PK1K2_block = tl.exp(K1K2_block)

        dV1_acc += tl.sum(
            PK1K2_block[:, :, None]
            * acc_dOv1[None, :, :]
            * tl.trans(V2_block)[None, :, :],
            1,
        )

        # Compute dV2 contribution for this k2 block
        dV2_contrib = tl.sum(
            PK1K2_block[:, :, None]
            * acc_dOv1[None, :, :]
            * tl.trans(V1_block)[:, None, :],
            0,
        )

        # Store accumulated dV2 contribution
        tl.atomic_add(
            dV2
            + offset_dV2
            + offs_k2[:, None] * dV2_strideS
            + offs_dim[None, :] * dV2_strideD,
            dV2_contrib,
            mask=mask_k2[:, None] & (offs_dim[None, :] < HEAD_DIM),
        )

        acc_QK20 = acc_QK2O * tl.trans(V2_block)
        if not convert_to_float32:
            acc_QK20 = acc_QK20.to(V1_block.dtype)
        if input_precision is not None:
            dS_k1 = tl.dot(acc_QK20, V1_block, input_precision=input_precision)
        else:
            dS_k1 = tl.dot(acc_QK20, V1_block)

        dS_k1 -= acc_QDK2_sub[:, None]

        # dK1 += PK1K2_block * dS_k1 @ K2_block
        if not convert_to_float32:
            dS_k1 = dS_k1.to(K1_block.dtype)
            PK1K2_block = PK1K2_block.to(K1_block.dtype)
        if input_precision is not None:
            dK1_contrib = (
                tl.dot(
                    PK1K2_block * tl.trans(dS_k1),
                    K2_block,
                    input_precision=input_precision,
                )
                * softmax_scale
            )
        else:
            dK1_contrib = (
                tl.dot(PK1K2_block * tl.trans(dS_k1), K2_block) * softmax_scale
            )
        dK1_acc += dK1_contrib

        if input_precision is not None:
            dK2_contrib = (
                tl.dot(
                    tl.trans(PK1K2_block * tl.trans(dS_k1)),
                    K1_block,
                    input_precision=input_precision,
                )
                * softmax_scale
            )
        else:
            dK2_contrib = (
                tl.dot(tl.trans(PK1K2_block * tl.trans(dS_k1)), K1_block)
                * softmax_scale
            )

        # Store accumulated dK2 contribution using atomic_add
        tl.atomic_add(
            dK2
            + batch_id * dK2_strideB
            + head_id * dK2_strideH
            + offs_k2[:, None] * dK2_strideS
            + offs_dim[None, :] * dK2_strideD,
            dK2_contrib,
            mask=mask_k2[:, None] & (offs_dim[None, :] < HEAD_DIM),
        )

    # Store results
    tl.store(
        dK1
        + offset_dK1
        + offs_k1[:, None] * dK1_strideS
        + offs_dim[None, :] * dK1_strideD,
        dK1_acc,
        mask=(offs_k1[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
    )

    tl.store(
        dV1
        + offset_dV1
        + offs_k1[:, None] * dV1_strideS
        + offs_dim[None, :] * dV1_strideD,
        dV1_acc,
        mask=(offs_k1[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
    )

    # dV2 and dK2 are stored inside the k2 loop using atomic_add


def tritt_bwd_dk1_dv1(
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
    Wrapper function for backward dK1, dV1, dV2, and dK2 computation.

    Args:
        Q, K1, K2, V1, V2: Input tensors
        D: Preprocessing result from forward pass
        softmax_scale: Scaling factor for attention
        dO: Gradient of output
        M: Logits normalization from forward pass
        causal: Whether to use causal masking
        k_diff: Maximum distance for k-masking

    Returns:
        dK1, dV1, dV2, dK2: Gradients w.r.t. K1, V1, V2, and K2
    """
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

    # Apply F.silu(V1) in the main function as instructed
    silu_V1 = F.silu(V1)

    # Make tensors contiguous
    Q = Q.contiguous()
    K1 = K1.contiguous()
    K2 = K2.contiguous()
    V1 = V1.contiguous()
    V2 = V2.contiguous()
    D = D.contiguous()
    dO = dO.contiguous()
    M = M.contiguous()

    # Initialize output tensors
    dK1 = torch.zeros_like(K1)
    dV1 = torch.zeros_like(V1)
    dV2 = torch.zeros_like(V2)
    dK2 = torch.zeros_like(K2)

    # Grid configuration
    grid = (BATCH_SIZE * NUM_HEADS, triton.cdiv(SEQ_LEN, 32))

    # Block sizes
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_KV = 32

    # Get strides
    q_strides = Q.stride()
    k1_strides = K1.stride()
    k2_strides = K2.stride()
    v1_strides = V1.stride()
    v2_strides = V2.stride()
    d_strides = D.stride()
    dO_strides = dO.stride()
    m_strides = M.stride()
    dK1_strides = dK1.stride()
    dV1_strides = dV1.stride()
    dV2_strides = dV2.stride()
    dK2_strides = dK2.stride()

    # Launch kernel
    _tritt_bwd_dk1_dv1[grid](
        Q,
        K1,
        K2,
        silu_V1,
        V2,
        D,
        softmax_scale,
        dO,
        M,
        dK1,
        dV1,
        dV2,
        dK2,
        q_strides[0],
        q_strides[1],
        q_strides[2],
        q_strides[3],
        k1_strides[0],
        k1_strides[1],
        k1_strides[2],
        k1_strides[3],
        k2_strides[0],
        k2_strides[1],
        k2_strides[2],
        k2_strides[3],
        v1_strides[0],
        v1_strides[1],
        v1_strides[2],
        v1_strides[3],
        v2_strides[0],
        v2_strides[1],
        v2_strides[2],
        v2_strides[3],
        d_strides[0],
        d_strides[1],
        d_strides[2],
        dO_strides[0],
        dO_strides[1],
        dO_strides[2],
        dO_strides[3],
        m_strides[0],
        m_strides[1],
        m_strides[2],
        dK1_strides[0],
        dK1_strides[1],
        dK1_strides[2],
        dK1_strides[3],
        dV1_strides[0],
        dV1_strides[1],
        dV1_strides[2],
        dV1_strides[3],
        dV2_strides[0],
        dV2_strides[1],
        dV2_strides[2],
        dV2_strides[3],
        dK2_strides[0],
        dK2_strides[1],
        dK2_strides[2],
        dK2_strides[3],
        SEQ_LEN,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        HEAD_DIM,
        NUM_HEADS,
        causal,
        k_diff,
        convert_to_float32,
        input_precision,
    )
    sigmoid_v1 = torch.sigmoid(V1)
    silu_derivative = sigmoid_v1 * (1 + V1 * (1 - sigmoid_v1))
    dV1 = dV1 * silu_derivative

    return dK1, dV1, dV2, dK2
