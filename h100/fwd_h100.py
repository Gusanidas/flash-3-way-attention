"""H100 (Hopper, sm_90a) tuned forward kernel for triple_dot_product trittention.

Same math and online-softmax bookkeeping as the B200 kernel in
b200/fwd_b200.py (QK2 precompute -> dense matmuls, base-2 exponentials,
interior-tile fast path; saved M/L semantics identical to the baseline so the
main package's backward kernels are reused unchanged). What is Hopper-specific:

1. **TMA descriptor loads** (USE_TMA constexpr). K1/K2/V1/V2 tiles are loaded
   through device-side tensor descriptors (`tl.make_tensor_descriptor`), which
   lower to the Hopper TMA engine: the address math leaves the SM, loads are
   issued as bulk async copies, and out-of-bounds rows are zero-filled by
   hardware (replacing the bounds masks on the plain-load path). This is the
   headline sm_90 memory feature; requires `triton.set_allocator` on the host
   (done in the wrapper).

2. **Hopper-shaped autotune space.** wgmma wants the matmul M dim (here
   BQ*BKV2) in multiples of 64 per warpgroup, and Hopper leans harder on
   deep software pipelining than Blackwell — the space probes num_stages 2-4
   and num_warps 4 (one warpgroup) as well as 8.

Requires: contiguous inputs (the wrapper enforces this), last-dim stride 1.
"""

import os

import triton
import triton.language as tl


def get_h100_autotune_configs():
    forced = os.environ.get("H100_FORCE_CONFIG")
    if forced:
        bq, bkv2, bkv1, nw, ns = (int(x) for x in forced.split(","))
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_Q": bq,
                    "BLOCK_SIZE_KV2": bkv2,
                    "BLOCK_SIZE_KV1": bkv1,
                },
                num_stages=ns,
                num_warps=nw,
            )
        ]
    # Hopper-oriented space. Rows of the main matmul are BQ*BKV2 — kept at
    # 128/256 (multiples of the wgmma M tile). BKV1 is the axis allowed to
    # grow. Deeper stage counts than the B200 space: Hopper hides HBM latency
    # through pipelining where Blackwell has more L2/SM bandwidth headroom.
    # (8,16,64,w4,ns3) won the first sweep (rows = 128 = one wgmma M-tile for a
    # single warpgroup, deep-ish pipeline); this space probes its neighborhood
    # plus small tiles for narrow windows and the B200 winner for cross-arch.
    shapes = [
        (8, 16, 64, 4, 3),
        (8, 16, 64, 4, 4),
        (8, 16, 64, 8, 3),
        (4, 16, 64, 4, 3),
        (8, 8, 64, 4, 3),
        (8, 32, 64, 4, 3),
        (8, 16, 128, 4, 3),
        (16, 16, 64, 4, 3),
        (16, 16, 16, 4, 3),
        (16, 16, 64, 8, 2),
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
    configs=get_h100_autotune_configs(),
    key=["HEAD_DIM", "SEQ_LEN", "USE_TMA", "convert_to_float32"],
)
@triton.jit
def _tritt_fwd_h100(
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
    USE_TMA: tl.constexpr,
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
    # Fold the FULL softmax scale AND log2(e) into Q once; the kernel works in
    # base-2 exponentials and the stored M is converted back to ln at the end.
    # Form scores in fp32 even when values remain in the input dtype.  The
    # Q*K2 pre-product otherwise rounds before the K1 reduction and diverges
    # noticeably from the reference for the common D=64 fp16 case.
    Q_scores = Q_block.to(tl.float32) * (softmax_scale * LOG2E)

    if USE_TMA:
        # Per-(batch, head) 2D descriptors over the (seq, dim) plane. TMA
        # zero-fills out-of-bounds rows, so no bounds masks are needed on
        # these loads. Last dim must be stride-1 (wrapper makes inputs
        # contiguous).
        desc_k2 = tl.make_tensor_descriptor(
            K2 + qvk_offset,
            shape=[SEQ_LEN, HEAD_DIM],
            strides=[stride_K2_seq, 1],
            block_shape=[BLOCK_SIZE_KV2, HEAD_DIM],
        )
        desc_v2 = tl.make_tensor_descriptor(
            V2 + qvk_offset,
            shape=[SEQ_LEN, HEAD_DIM],
            strides=[stride_V2_seq, 1],
            block_shape=[BLOCK_SIZE_KV2, HEAD_DIM],
        )
        desc_k1 = tl.make_tensor_descriptor(
            K1 + qvk_offset,
            shape=[SEQ_LEN, HEAD_DIM],
            strides=[stride_K1_seq, 1],
            block_shape=[BLOCK_SIZE_KV1, HEAD_DIM],
        )
        desc_v1 = tl.make_tensor_descriptor(
            V1 + qvk_offset,
            shape=[SEQ_LEN, HEAD_DIM],
            strides=[stride_V1_seq, 1],
            block_shape=[BLOCK_SIZE_KV1, HEAD_DIM],
        )

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

        if USE_TMA:
            K2_block = desc_k2.load([start_kv_2, 0])
            V2_block = desc_v2.load([start_kv_2, 0])
        else:
            k2_block_ptr = (
                K2
                + qvk_offset
                + offs_kv2[:, None] * stride_K2_seq
                + offs_d[None, :] * stride_K2_dim
            )
            K2_block = tl.load(k2_block_ptr, mask=mask_kv2[:, None], other=0.0)
            v2_block_ptr = (
                V2
                + qvk_offset
                + offs_kv2[:, None] * stride_V2_seq
                + offs_d[None, :] * stride_V2_dim
            )
            V2_block = tl.load(v2_block_ptr, mask=mask_kv2[:, None], other=0.0)
        if convert_to_float32:
            K2_block = K2_block.to(tl.float32)
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

            if USE_TMA:
                # (BKV1, D) tile; transposed for the dot below.
                K1_block = tl.trans(desc_k1.load([start_kv_1, 0]))
            else:
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

            if USE_TMA:
                V1_block = desc_v1.load([start_kv_1, 0])
            else:
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
