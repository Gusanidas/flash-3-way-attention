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
            for bq in [16, 32, 64, 128]
            for bkv in [16, 32, 64, 128]
            for ns in [1, 2, 3]
            for nw in [1, 2, 4, 8]
        ]


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
    k_diff,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv_1: tl.constexpr,
    offs_kv_2: tl.constexpr,
    convert_to_float32: tl.constexpr,
    input_precision: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    # Round the diagonal boundary DOWN to a multiple of BLOCK_SIZE_KV so that the
    # off-diagonal (STAGE 1) and diagonal (STAGE 2) passes tile K2 exactly once each,
    # with no gap and no overlap, for ANY (BLOCK_SIZE_Q, BLOCK_SIZE_KV) combination.
    # STAGE 1 handles the K2 blocks strictly below the diagonal (all k2 < q, so no
    # q>=k2 mask needed); STAGE 2 handles the single straddling block plus everything
    # up to the block diagonal, applying the q>=k2 mask.
    diag_start = (block_index_q * BLOCK_SIZE_Q // BLOCK_SIZE_KV) * BLOCK_SIZE_KV
    if STAGE == 1:
        lo, hi = 0, diag_start
    elif STAGE == 2:
        lo, hi = diag_start, (block_index_q + 1) * BLOCK_SIZE_Q
    else:
        # Non-causal attention (process all tokens)
        lo, hi = 0, SEQ_LEN

    for start_kv_2 in range(lo, hi, BLOCK_SIZE_KV):
        # lo is always a multiple of BLOCK_SIZE_KV (0, diag_start, or 0), so every
        # start_kv_2 is too -- the alignment hint below is now always valid.
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV)

        lo_1_k_diff = BLOCK_SIZE_KV * (
            tl.maximum(0, start_kv_2 - k_diff) // BLOCK_SIZE_KV
        )
        if STAGE < 3:
            k1_hi = start_kv_2 + BLOCK_SIZE_KV
            lo_1 = tl.maximum(lo_1_k_diff, 0)
        else:
            k1_hi = hi
            lo_1 = 0

        current_offs_k2_seq = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        current_k2_block_ptr = (
            K2
            + qvk_offset
            + tl.arange(0, HEAD_DIM)[:, None] * stride_K2_dim
            + current_offs_k2_seq[None, :] * stride_K2_seq
        )
        mask_k2 = current_offs_k2_seq < SEQ_LEN
        K2_block = tl.load(current_k2_block_ptr, mask=mask_k2[None, :], other=0.0)

        acc_k1k2_v1 = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
        # Per-COLUMN running max of the k1k2 leg (two-level online LSE). A
        # per-block scalar max lets columns far below the block max underflow
        # (l_jk -> 0 -> 0/0 NaN); with a per-column max every valid column keeps
        # l_jk >= 1.
        m_jk = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32) - 1.0e7
        l_jk = tl.zeros([BLOCK_SIZE_KV], dtype=tl.float32)

        for start_kv_1 in range(lo_1, k1_hi, BLOCK_SIZE_KV):
            start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV)

            current_offs_k1_seq = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
            current_k1_block_ptr = (
                K1
                + qvk_offset
                + current_offs_k1_seq[:, None] * stride_K1_seq
                + tl.arange(0, HEAD_DIM)[None, :] * stride_K1_dim
            )
            mask_k1 = current_offs_k1_seq < SEQ_LEN
            K1_block = tl.load(current_k1_block_ptr, mask=mask_k1[:, None], other=0.0)

            current_offs_v1_seq = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV)
            current_v1_block_ptr = (
                V1
                + qvk_offset
                + current_offs_v1_seq[:, None] * stride_V1_seq
                + tl.arange(0, HEAD_DIM)[None, :] * stride_V1_dim
            )
            mask_v1 = current_offs_v1_seq < SEQ_LEN
            V1_block = tl.load(current_v1_block_ptr, mask=mask_v1[:, None], other=0.0)
            if input_precision is not None:
                K1K2_block = tl.dot(K1_block, K2_block, input_precision=input_precision)
            else:
                K1K2_block = tl.dot(K1_block, K2_block)

            mask_k1_k2 = mask_k1[:, None] & mask_k2[None, :]
            if STAGE < 3:
                mask_k1_lt_k2 = (
                    current_offs_k1_seq[:, None] <= current_offs_k2_seq[None, :]
                )  # [BLOCK_SIZE_KV, BLOCK_SIZE_KV]
                mask_k1_k2_k_diff = (
                    current_offs_k1_seq[:, None] + k_diff
                    >= current_offs_k2_seq[None, :]
                )  # [BLOCK_SIZE_KV, BLOCK_SIZE_KV]
                mask_k1_lt_k2 = mask_k1_lt_k2 & mask_k1_k2_k_diff
                mask_k1_k2 = mask_k1_k2 & mask_k1_lt_k2

            K1K2_block = K1K2_block * softmax_scale + tl.where(mask_k1_k2, 0, -1.0e7)

            new_m_jk = tl.maximum(m_jk, tl.max(K1K2_block, axis=0))
            K1K2_block -= new_m_jk[None, :]

            P_k1k2_block = tl.math.exp(K1K2_block)
            v1_f32 = V1_block.to(tl.float32)
            v1_f32 = v1_f32 * tl.sigmoid(v1_f32)
            V1_block = v1_f32.to(V1_block.type.element_ty)
            alpha = tl.math.exp(m_jk - new_m_jk)
            P_k1k2_block = P_k1k2_block.to(V1_block.type.element_ty)
            if input_precision is not None:
                acc_k1k2_v1 = acc_k1k2_v1 * alpha[:, None] + tl.dot(
                    tl.trans(P_k1k2_block), V1_block, input_precision=input_precision
                )
            else:
                acc_k1k2_v1 = acc_k1k2_v1 * alpha[:, None] + tl.dot(
                    tl.trans(P_k1k2_block), V1_block
                )
            l_jk = l_jk * alpha + tl.sum(P_k1k2_block, axis=0)
            m_jk = new_m_jk

        current_offs_v2_seq = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV)
        current_v2_block_ptr = (
            V2
            + qvk_offset
            + current_offs_v2_seq[:, None] * stride_V2_seq
            + tl.arange(0, HEAD_DIM)[None, :] * stride_V2_dim
        )
        mask_v2 = current_offs_v2_seq < SEQ_LEN
        V2_block = tl.load(current_v2_block_ptr, mask=mask_v2[:, None], other=0.0)

        # Keep the k1k2 accumulators in per-column max-subtracted form. Instead of
        # multiplying exp(m_jk) back in here -- which would materialise raw
        # exp(scale*K1.K2) and overflow -- m_jk is folded additively into the OUTER
        # softmax logit below, so the accumulators stay bounded in [0, BLOCK].
        o1 = acc_k1k2_v1 * V2_block

        if input_precision is not None:
            QK2_block = tl.dot(Q_block, K2_block, input_precision=input_precision)
        else:
            QK2_block = tl.dot(Q_block, K2_block)
        mask_q_k2 = (offs_q[:, None] < SEQ_LEN) & mask_k2[None, :]
        if STAGE == 2:
            mask_q_gt_k2 = offs_q[:, None] >= current_offs_k2_seq[None, :]
            mask_q_k2 = mask_q_k2 & mask_q_gt_k2

        # Combined outer logit = scale*Q.K2 + m_jk[t] (per-column log-scale of the
        # k1k2 leg subtracted out of o1/l_jk). l_jk and o1 are in per-column
        # max-subtracted form, so P_qk * l_jk and P_qk @ o1 reconstruct the exact
        # softmax over (s, t) pairs with no column ever underflowing.
        QK2_block = (
            QK2_block * softmax_scale
            + m_jk[None, :]
            + tl.where(mask_q_k2, 0, -1.0e7)
        )

        m_ij = tl.maximum(m_i, tl.max(QK2_block, axis=1))
        QK2_block -= m_ij[:, None]

        P_qk = tl.math.exp(QK2_block)
        if input_precision is not None:
            o_out = tl.dot(P_qk, o1, input_precision=input_precision)
        else:
            o_out = tl.dot(P_qk, o1)
        l_ij = tl.sum(P_qk * l_jk[None, :], 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        O_block = O_block * alpha[:, None] + o_out
        m_i = m_ij

    return O_block, l_i, m_i


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
    STAGE: tl.constexpr,
    k_diff: tl.constexpr,
    convert_to_float32: tl.constexpr,
    input_precision: tl.constexpr,
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

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    q_block_ptr = (
        Q
        + qvk_offset
        + offs_q[:, None] * stride_Q_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_Q_dim
    )
    mask_q = offs_q < SEQ_LEN

    o_block_ptr = (
        O
        + qvk_offset
        + offs_q[:, None] * stride_O_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_O_dim
    )

    offs_kv_1 = tl.arange(0, BLOCK_SIZE_KV)
    offs_kv_2 = tl.arange(0, BLOCK_SIZE_KV)

    Q_block = tl.load(q_block_ptr, mask=mask_q[:, None], other=0.0)

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - 1.0e7
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # For causal attention (STAGE=3), we split processing into two parts
    if STAGE == 1 or STAGE == 3:
        # Process blocks to the left of the diagonal (without masking)
        O_block, l_i, m_i = _tritt_fwd_inner(
            O_block=O_block,
            l_i=l_i,
            m_i=m_i,
            Q_block=Q_block,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            qvk_offset=qvk_offset,
            stride_K1_seq=stride_K1_seq,
            stride_K1_dim=stride_K1_dim,
            stride_K2_seq=stride_K2_seq,
            stride_K2_dim=stride_K2_dim,
            stride_V1_seq=stride_V1_seq,
            stride_V1_dim=stride_V1_dim,
            stride_V2_seq=stride_V2_seq,
            stride_V2_dim=stride_V2_dim,
            softmax_scale=softmax_scale,
            block_index_q=block_index_q,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=4 - STAGE,  # STAGE = 1 for left of diagonal
            offs_q=offs_q,
            offs_kv_1=offs_kv_1,
            offs_kv_2=offs_kv_2,
            k_diff=k_diff,
            convert_to_float32=convert_to_float32,
            input_precision=input_precision,
        )

    if STAGE == 3:
        # Process the diagonal block (with masking)
        O_block, l_i, m_i = _tritt_fwd_inner(
            O_block=O_block,
            l_i=l_i,
            m_i=m_i,
            Q_block=Q_block,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            qvk_offset=qvk_offset,
            stride_K1_seq=stride_K1_seq,
            stride_K1_dim=stride_K1_dim,
            stride_K2_seq=stride_K2_seq,
            stride_K2_dim=stride_K2_dim,
            stride_V1_seq=stride_V1_seq,
            stride_V1_dim=stride_V1_dim,
            stride_V2_seq=stride_V2_seq,
            stride_V2_dim=stride_V2_dim,
            softmax_scale=softmax_scale,
            block_index_q=block_index_q,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=2,  # STAGE = 2 for diagonal block
            offs_q=offs_q,
            offs_kv_1=offs_kv_1,
            offs_kv_2=offs_kv_2,
            k_diff=k_diff,
            convert_to_float32=convert_to_float32,
            input_precision=input_precision,
        )

    m_i += tl.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    m_mask = offs_q < SEQ_LEN
    tl.store(m_ptrs, m_i, mask=m_mask)

    if convert_to_float32:
        tl.store(o_block_ptr, O_block.to(tl.float32), mask=mask_q[:, None])
    else:
        tl.store(o_block_ptr, O_block.to(O.type.element_ty), mask=mask_q[:, None])
