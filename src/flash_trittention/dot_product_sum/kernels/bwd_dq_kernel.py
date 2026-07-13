import triton
import triton.language as tl

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
    m_q = tl.load(
        M + (offset_m + offs_q * M_strideS), mask=offs_q < SEQ_LEN, other=-1.0e9
    )
    if convert_to_float32:
        m_q = m_q.to(tl.float32)

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

        acc_k1k2_v1 = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
        acc_pk2_k1 = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32)
        # Per-column (k2) running max of the k1k2 leg. It is subtracted from the
        # k1k2 exponent here and added back into the QK2 exponent below, so it
        # cancels exactly while keeping both exp factors <= 1 (M >= scale*Q.K2 +
        # mk[t] for every valid (q, t)).
        mk = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32) - 1.0e9

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

            K1K2_block = K1K2_block * softmax_scale

            new_mk = tl.maximum(
                mk, tl.max(tl.where(mask_k1_k2, K1K2_block, -1.0e9), axis=0)
            )
            K1K2_block = tl.where(mask_k1_k2, K1K2_block, float("-inf"))
            P_k1k2_block = tl.math.exp(K1K2_block - new_mk[None, :])
            alpha_k = tl.math.exp(mk - new_mk)
            P_k1k2_block = P_k1k2_block.to(dO_block.dtype)
            if input_precision is not None:
                acc_k1k2_v1 = acc_k1k2_v1 * alpha_k[:, None] + tl.trans(
                    tl.dot(V1_block, P_k1k2_block, input_precision=input_precision)
                )
            else:
                acc_k1k2_v1 = acc_k1k2_v1 * alpha_k[:, None] + tl.trans(
                    tl.dot(V1_block, P_k1k2_block)
                )
            acc_pk2_k1 = acc_pk2_k1 * alpha_k + tl.sum(P_k1k2_block, 0)
            mk = new_mk

        if input_precision is not None:
            QK2_block = tl.dot(Q_block, K2_block, input_precision=input_precision)
        else:
            QK2_block = tl.dot(Q_block, K2_block)
        mask_q_k2 = (offs_q[:, None] < SEQ_LEN) & mask_k2[None, :]
        if CAUSAL:
            mask_q_gt_k2 = offs_q[:, None] >= current_offs_k2_seq[None, :]
            mask_q_k2 = mask_q_k2 & mask_q_gt_k2

        QK2_block = QK2_block * softmax_scale - m_q[:, None] + mk[None, :]
        QK2_block = tl.where(mask_q_k2, QK2_block, float("-inf"))
        P_qk2 = tl.math.exp(QK2_block)
        # P_qk = P_qk * tl.where(mask_q_k2, 1, 0)
        current_offs_v2_seq = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        current_v2_block_ptr = (
            V2
            + offset_v2
            + current_offs_v2_seq[:, None] * v2_strideS
            + tl.arange(0, HEAD_DIM)[None, :] * v2_strideD
        )
        mask_v2 = current_offs_v2_seq < SEQ_LEN
        V2_block = tl.load(current_v2_block_ptr, mask=mask_v2[:, None], other=0.0)
        if convert_to_float32:
            V2_block = V2_block.to(tl.float32)
        dS_v1v2 = acc_k1k2_v1 * V2_block
        if not convert_to_float32:
            dS_v1v2 = dS_v1v2.to(Q_block.dtype)
            P_qk2 = P_qk2.to(Q_block.dtype)
        if input_precision is not None:
            dS_left = P_qk2 * tl.dot(
                dO_block, tl.trans(dS_v1v2), input_precision=input_precision
            )
        else:
            dS_left = P_qk2 * tl.dot(dO_block, tl.trans(dS_v1v2))
        dS_right = P_qk2 * D_block[:, None] * acc_pk2_k1[None, :]

        dS = dS_left - dS_right
        if not convert_to_float32:
            dS = dS.to(K2_block.dtype)
        if input_precision is not None:
            dQ_block += softmax_scale * tl.dot(
                dS, tl.trans(K2_block), input_precision=input_precision
            )
        else:
            dQ_block += softmax_scale * tl.dot(dS, tl.trans(K2_block))

        # Accumulate dK2 gradients from Q-K2 path
        if input_precision is not None:
            acc_dK2_from_qk = softmax_scale * tl.dot(
                tl.trans(dS), Q_block, input_precision=input_precision
            )
        else:
            acc_dK2_from_qk = softmax_scale * tl.dot(tl.trans(dS), Q_block)

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
