"""B200 (sm_100a) tuned forward kernel for triple_dot_product trittention.

Structural changes vs the baseline kernel in
src/flash_trittention/triple_dot_product/kernels/fwd.py:

1. The baseline materializes the outer product k1[j,:] * K2[k,:] as a
   (HEAD_DIM, BKV*BKV) tensor inside the inner kv1 loop — once per
   (kv1, kv2) block pair. Here we instead precompute
   QK2[(q,k), d] = Q[q, d] * K2[k, d] ONCE per kv2 block (Q is pre-scaled by
   softmax_scale, so no sqrt-scale split is needed), and each inner
   iteration becomes a single dense matmul
       P[(q,k), j] = QK2 @ K1^T      -- (BQ*BKV2, D) @ (D, BKV1)
   which is tensor-core shaped and lets BLOCK_KV1 grow independently.

2. The value side avoids the v1⊗v2 outer product the same way:
       T[(q,k), d] = exp_P @ silu(V1)  -- (BQ*BKV2, BKV1) @ (BKV1, D)
       O[q, d]    += sum_k T[q, k, d] * V2[k, d]
   The V2 multiply-reduce happens once per inner iteration on a
   (BQ, BKV2, D) tile, which is cheap next to the dots.

The online-softmax bookkeeping (m_i / l_i, phantom init m=-1000, l=1) and the
saved M = m + log(l) semantics are IDENTICAL to the baseline, so the existing
backward kernels are reused unchanged.
"""

import os

import triton
import triton.language as tl


def get_b200_autotune_configs():
    forced = os.environ.get("B200_FORCE_CONFIG")
    if forced:
        bq, bkv2, bkv1, nw = (int(x) for x in forced.split(","))
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_Q": bq,
                    "BLOCK_SIZE_KV2": bkv2,
                    "BLOCK_SIZE_KV1": bkv1,
                },
                num_stages=3,
                num_warps=nw,
            )
        ]
    # Blackwell-oriented space. Register pressure is dominated by the
    # (BQ*BKV2, BKV1) logits tile and the (BQ*BKV2, D) QK2 tile, so BQ*BKV2 is
    # kept <= 512 and BKV1 is the axis allowed to grow.
    # (BQ, BKV2, BKV1, warps, stages). BKV2 does not participate in the tl.dot
    # shapes (rows are BQ*BKV2), so it can shrink below 16 to trade for a bigger
    # BQ. (16,16,64,w8) won the full-causal sweep on B200; stage variants and a
    # 128-wide KV1 probe around it, plus small tiles for narrow-window cases.
    shapes = [
        (16, 16, 16, 4, 3),
        (16, 16, 16, 8, 2),
        (16, 16, 32, 8, 3),
        (16, 16, 64, 8, 2),
        (16, 16, 64, 8, 3),
        (16, 16, 64, 8, 4),
        (16, 16, 128, 8, 3),
        (32, 16, 64, 8, 3),
        (32, 8, 128, 8, 3),
        (64, 8, 64, 8, 3),
    ]
    return [
        triton.Config(
            {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV2": bkv2, "BLOCK_SIZE_KV1": bkv1},
            num_stages=ns,
            num_warps=nw,
        )
        for bq, bkv2, bkv1, nw, ns in shapes
    ]


@triton.jit
def _causal_mask_qk2k1(
    kk_left,
    kk_right,
    k1_window,
    k2_window,
    offs_q,
    offs_kv1,
    offs_kv2,
):
    """Causal mask in (Q, kv2, kv1) layout — same convention as
    utils.get_causal_mask, duplicated here so this file is self-contained for
    broker shipping."""
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


@triton.autotune(
    configs=get_b200_autotune_configs(),
    key=["HEAD_DIM", "SEQ_LEN", "convert_to_float32"],
)
@triton.jit
def _tritt_fwd_b200(
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
    kk_left: tl.constexpr,
    kk_right: tl.constexpr,
    k1_window: tl.constexpr,
    k2_window: tl.constexpr,
    causal: tl.constexpr,
    convert_to_float32: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    BLOCK_SIZE_KV1: tl.constexpr,
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
    offs_d = tl.arange(0, HEAD_DIM)
    mask_q = offs_q < SEQ_LEN

    q_block_ptr = (
        Q
        + qvk_offset
        + offs_q[:, None] * stride_Q_seq
        + offs_d[None, :] * stride_Q_dim
    )
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    Q_block = tl.load(q_block_ptr, mask=mask_q[:, None], other=0.0)
    if convert_to_float32:
        Q_block = Q_block.to(tl.float32)
    # Fold the FULL softmax scale AND log2(e) into Q once: the kernel then works
    # in base-2 exponentials (exp2 is the native SFU op; exp lowers to
    # mul+exp2 per element). m_i/l_i bookkeeping stays exactly equivalent; the
    # stored M is converted back to ln units at the end.
    # Form scores in fp32 even when values remain in the input dtype.  The
    # Q*K2 pre-product otherwise rounds before the K1 reduction and diverges
    # noticeably from the reference for the common D=64 fp16 case.
    Q_scores = Q_block.to(tl.float32) * (softmax_scale * LOG2E)

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - 1000.0 * LOG2E
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # kv2 interval for this q block (same logic as intervals.interval_k2_from_q)
    if causal:
        end_q = block_index_q * BLOCK_SIZE_Q + BLOCK_SIZE_Q
        hi_k2 = tl.minimum(end_q, SEQ_LEN)
        lo_k2 = tl.maximum(0, block_index_q * BLOCK_SIZE_Q - k2_window)
        lo_k2 = (lo_k2 // BLOCK_SIZE_KV2) * BLOCK_SIZE_KV2
    else:
        lo_k2, hi_k2 = 0, SEQ_LEN

    for start_kv_2 in range(lo_k2, hi_k2, BLOCK_SIZE_KV2):
        start_kv_2 = tl.multiple_of(start_kv_2, BLOCK_SIZE_KV2)
        offs_kv2 = start_kv_2 + tl.arange(0, BLOCK_SIZE_KV2)
        mask_kv2 = offs_kv2 < SEQ_LEN

        k2_block_ptr = (
            K2
            + qvk_offset
            + offs_kv2[:, None] * stride_K2_seq
            + offs_d[None, :] * stride_K2_dim
        )
        K2_block = tl.load(k2_block_ptr, mask=mask_kv2[:, None], other=0.0)
        if convert_to_float32:
            K2_block = K2_block.to(tl.float32)

        v2_block_ptr = (
            V2
            + qvk_offset
            + offs_kv2[:, None] * stride_V2_seq
            + offs_d[None, :] * stride_V2_dim
        )
        V2_block = tl.load(v2_block_ptr, mask=mask_kv2[:, None], other=0.0)
        if convert_to_float32:
            V2_block = V2_block.to(tl.float32)

        # QK2[(q,k), d] = Q[q,d] (pre-scaled) * K2[k,d] — once per kv2 block.
        QK2 = tl.reshape(
            Q_scores[:, None, :] * K2_block.to(tl.float32)[None, :, :],
            (BLOCK_SIZE_Q * BLOCK_SIZE_KV2, HEAD_DIM),
        )

        # kv1 interval for this (q, kv2) block pair (same as intervals.interval_k1)
        if causal:
            # mask convention: j >= k - kk_left and j <= k + kk_right
            hi_k1 = tl.minimum(
                tl.minimum(end_q, start_kv_2 + BLOCK_SIZE_KV2 + kk_right), SEQ_LEN
            )
            lo_k1 = tl.maximum(
                tl.maximum(0, block_index_q * BLOCK_SIZE_Q - k1_window),
                start_kv_2 - kk_left,
            )
            lo_k1 = (lo_k1 // BLOCK_SIZE_KV1) * BLOCK_SIZE_KV1
            lo_k1 = tl.maximum(lo_k1, 0)
        else:
            lo_k1, hi_k1 = 0, SEQ_LEN

        for start_kv_1 in range(lo_k1, hi_k1, BLOCK_SIZE_KV1):
            start_kv_1 = tl.multiple_of(start_kv_1, BLOCK_SIZE_KV1)
            offs_kv1 = start_kv_1 + tl.arange(0, BLOCK_SIZE_KV1)
            mask_kv1 = offs_kv1 < SEQ_LEN

            k1_block_ptr = (
                K1
                + qvk_offset
                + offs_d[:, None] * stride_K1_dim
                + offs_kv1[None, :] * stride_K1_seq
            )
            K1_block = tl.load(k1_block_ptr, mask=mask_kv1[None, :], other=0.0)
            if convert_to_float32:
                K1_block = K1_block.to(tl.float32)

            # P[(q,k), j] — one dense matmul per inner iteration (base-2 logits).
            P = tl.dot(QK2, K1_block.to(tl.float32))
            P = tl.reshape(P, (BLOCK_SIZE_Q, BLOCK_SIZE_KV2, BLOCK_SIZE_KV1))

            # Fast path: skip mask construction entirely when the whole
            # (q, kv2, kv1) tile is in-bounds and causally valid. Scalar
            # (non-divergent) conditions only — expressed as the OR of
            # violations so no scalar `not` is needed.
            q0 = block_index_q * BLOCK_SIZE_Q
            needs_mask = (
                (q0 + BLOCK_SIZE_Q > SEQ_LEN)
                | (start_kv_2 + BLOCK_SIZE_KV2 > SEQ_LEN)
                | (start_kv_1 + BLOCK_SIZE_KV1 > SEQ_LEN)
            )
            if causal:
                needs_mask = (
                    needs_mask
                    # violated iff some (q, j): q < j or q > j + k1_window
                    | (q0 < start_kv_1 + BLOCK_SIZE_KV1 - 1)
                    | (q0 + BLOCK_SIZE_Q - 1 > start_kv_1 + k1_window)
                    # violated iff some (q, k): q < k or q > k + k2_window
                    | (q0 < start_kv_2 + BLOCK_SIZE_KV2 - 1)
                    | (q0 + BLOCK_SIZE_Q - 1 > start_kv_2 + k2_window)
                    # violated iff some (j, k): j + kk_left < k or k + kk_right < j
                    | (start_kv_1 + kk_left < start_kv_2 + BLOCK_SIZE_KV2 - 1)
                    | (start_kv_2 + kk_right < start_kv_1 + BLOCK_SIZE_KV1 - 1)
                )

            if needs_mask:
                mask_qjk = (
                    mask_q[:, None, None]
                    & mask_kv2[None, :, None]
                    & mask_kv1[None, None, :]
                )
                if causal:
                    mask_qjk = mask_qjk & _causal_mask_qk2k1(
                        kk_left=kk_left,
                        kk_right=kk_right,
                        k1_window=k1_window,
                        k2_window=k2_window,
                        offs_q=offs_q,
                        offs_kv1=offs_kv1,
                        offs_kv2=offs_kv2,
                    )
                P = tl.where(mask_qjk, P, -float("inf"))

            new_m = tl.maximum(m_i, tl.max(tl.max(P, axis=2), axis=1))
            exp_P = tl.math.exp2(P - new_m[:, None, None])
            l_qjk = tl.sum(tl.sum(exp_P, axis=2), axis=1)

            v1_block_ptr = (
                V1
                + qvk_offset
                + offs_kv1[:, None] * stride_V1_seq
                + offs_d[None, :] * stride_V1_dim
            )
            V1_block = tl.load(v1_block_ptr, mask=mask_kv1[:, None], other=0.0)
            if convert_to_float32:
                V1_block = V1_block.to(tl.float32)

            exp_P_2d = tl.reshape(
                exp_P, (BLOCK_SIZE_Q * BLOCK_SIZE_KV2, BLOCK_SIZE_KV1)
            ).to(V1_block.dtype)

            # T[(q,k), d] = sum_j exp_P * silu(V1)[j, d]  (V1 pre-silu'd on host)
            T = tl.dot(exp_P_2d, V1_block)
            T = tl.reshape(T, (BLOCK_SIZE_Q, BLOCK_SIZE_KV2, HEAD_DIM))

            # contract kv2 against V2: O[q,d] += sum_k T[q,k,d] * V2[k,d]
            contrib = tl.sum(T * V2_block[None, :, :].to(T.dtype), axis=1)

            alpha = tl.math.exp2(m_i - new_m)
            l_i = alpha * l_i + l_qjk
            O_block = alpha[:, None] * O_block + contrib
            m_i = new_m

    O_block = O_block / l_i[:, None]

    # convert the base-2 running max back to ln units so the stored
    # M (+ log(L) in the wrapper) matches the baseline backward's convention
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i * LN2, mask=mask_q)
    l_ptrs = L + index_batch_head * SEQ_LEN + offs_q
    tl.store(l_ptrs, l_i, mask=mask_q)

    o_block_ptr = (
        O
        + qvk_offset
        + offs_q[:, None] * stride_O_seq
        + offs_d[None, :] * stride_O_dim
    )
    tl.store(o_block_ptr, O_block.to(O.type.element_ty), mask=mask_q[:, None])
