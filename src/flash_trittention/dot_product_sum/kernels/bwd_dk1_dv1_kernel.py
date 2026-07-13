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
def _tritt_bwd_dk_dv(
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
    Kernel for computing dK1, dV1
    """
    batch_head_id = tl.program_id(0)  # id for batch and head
    k1_v1_block_id = tl.program_id(1)  # id for k1 and v1 (same one)

    batch_id = batch_head_id // NUM_HEADS
    head_id = batch_head_id % NUM_HEADS

    # k1 is fixes, we loop over k2 and q.
    k1_start = k1_v1_block_id * BLOCK_SIZE_KV

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

    dK1_acc = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dV1_acc = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        k2_lo = k1_start
        k2_hi = tl.minimum(k1_start + k_diff + BLOCK_SIZE_KV + 1, SEQ_LEN)
    else:
        k2_lo = 0
        k2_hi = SEQ_LEN

    for k2_start_loop in range(k2_lo, k2_hi, BLOCK_SIZE_KV):
        k2_start_loop = tl.multiple_of(k2_start_loop, BLOCK_SIZE_KV)
        offs_k2 = k2_start_loop + tl.arange(0, BLOCK_SIZE_KV)
        mask_k2 = offs_k2 < SEQ_LEN

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

        # Recompute the k1k2 leg BEFORE the q-loop so we can derive a per-block max
        # (mk) that stabilizes exp(scale*K1.K2). mk is folded additively into the QK2
        # logit inside the q-loop and subtracted from the K1K2 logit here, so the
        # reconstructed probability PQK2 * PK1K2 = exp(scale*(Q.K2 + K1.K2) - M) is
        # unchanged (mk cancels) while neither factor overflows -- matching the
        # forward's convention where M = log(sum over (s,t) exp(scale*(Q.K2+K1.K2))).
        if input_precision is not None:
            K1K2_block = (
                tl.dot(K1_block, tl.trans(K2_block), input_precision=input_precision)
                * softmax_scale
            )
        else:
            K1K2_block = tl.dot(K1_block, tl.trans(K2_block)) * softmax_scale

        maskk1k2 = mask_k1[:, None] & mask_k2[None, :]
        if CAUSAL:
            mask_causal = offs_k1[:, None] <= offs_k2[None, :]
            mask_k1_k2_k_diff = offs_k1[:, None] + k_diff >= offs_k2[None, :]
            maskk1k2 = (maskk1k2 & mask_causal) & mask_k1_k2_k_diff

        # Per-COLUMN (k2) max for stabilization. Masked entries use a large finite
        # negative in the max (so a fully-masked column still yields a finite mk),
        # but -inf in the exponent (so masked entries give exactly 0). Per-column
        # (not per-block scalar) keeps every valid column's PK1K2 max at 1, so no
        # column can underflow, and M >= scale*Q.K2 + mk[t] bounds PQK2 <= 1.
        mk = tl.max(tl.where(maskk1k2, K1K2_block, -1.0e9), axis=0)
        K1K2_block = tl.where(maskk1k2, K1K2_block, float("-inf"))
        PK1K2_block = tl.exp(K1K2_block - mk[None, :])

        acc_QK2O = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
        # For the term that will be subtracted.
        acc_QDK2_sub = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32)

        # Define looping limits for q
        if CAUSAL:
            q_lo = k2_start_loop
            q_hi = SEQ_LEN
        else:
            q_lo = 0
            q_hi = SEQ_LEN

        for q_start in range(q_lo, q_hi, BLOCK_SIZE_Q):
            q_start = tl.multiple_of(q_start, BLOCK_SIZE_Q)
            offs_q = q_start + tl.arange(0, BLOCK_SIZE_Q)
            mask_q = offs_q < SEQ_LEN

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

            # QK2 = tl.dot(Q, K2) * softmax_scale - M + mk
            # The + mk folds in the k1k2-leg block max; it cancels against the -mk on
            # PK1K2 above, keeping PQK2 * PK1K2 exact while both factors stay bounded.
            if input_precision is not None:
                QK2_block = (
                    tl.dot(Q_block, tl.trans(K2_block), input_precision=input_precision)
                    * softmax_scale
                    - M_block[:, None]
                    + mk[None, :]
                )
            else:
                QK2_block = (
                    tl.dot(Q_block, tl.trans(K2_block)) * softmax_scale
                    - M_block[:, None]
                    + mk[None, :]
                )

            # Do necessary masking if causal
            maskqk = mask_q[:, None] & mask_k2[None, :]
            if CAUSAL:
                # distinct name from the (k1, k2)-shaped mask_causal above: reusing
                # the name makes Triton treat it as a loop-carried variable whose
                # shape changes (compile error)
                mask_causal_qk = offs_q[:, None] >= offs_k2[None, :]
                maskqk = maskqk & mask_causal_qk
            QK2_block = tl.where(maskqk, QK2_block, float("-inf"))

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

        # K1K2_block / PK1K2_block / mk were computed once above the q-loop.
        # acc_QK2O = sum_q trans(PQK2) @ dO is exactly the (formerly duplicated)
        # dO-weighted-by-PQK2 term used for both dV1 and dV2.
        dV1_acc += tl.sum(
            PK1K2_block[:, :, None]
            * acc_QK2O[None, :, :]
            * tl.trans(V2_block)[None, :, :],
            1,
        )

        # Compute dV2 contribution for this k2 block
        dV2_contrib = tl.sum(
            PK1K2_block[:, :, None]
            * acc_QK2O[None, :, :]
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
