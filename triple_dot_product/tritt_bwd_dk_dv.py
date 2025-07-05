import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _tritt_bwd_dv1_dk1(
    Q, K1, K2,
    V1, V2,
    D,
    softmax_scale,
    dO,
    dV1,
    dK1,
    M,
    q_strideB, q_strideH, q_strideS, q_strideD, # TODO: Pass only one set of strides
    # strides for K1
    k1_strideB, k1_strideH, k1_strideS, k1_strideD,
    # strides for K2
    k2_strideB, k2_strideH, k2_strideS, k2_strideD,
    # strides for V1
    v1_strideB, v1_strideH, v1_strideS, v1_strideD,
    # strides for V2
    v2_strideB, v2_strideH, v2_strideS, v2_strideD,
    # strides for D
    d_strideB, d_strideH, d_strideS,
    # strides for dO
    dO_strideB, dO_strideH, dO_strideS, dO_strideD,
    # strides for dV1
    dV1_strideB, dV1_strideH, dV1_strideS, dV1_strideD,
    # strides for dK1
    dK1_strideB, dK1_strideH, dK1_strideS, dK1_strideD,
    # strides for M
    M_strideB, M_strideH, M_strideS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr
):
    """
    Extended backward kernel to compute dV1 and dK1 in parallel.

    - dV1 is exactly as before.
    - dK1 = sum_{q,r} P_{q,t,r} * [ (dO_q * V1_t + dO_q * V2_r) - D_q ] * [Q_q + K2_r].
      Where:
        * P_{q,t,r} = exp( QK1K2 * softmax_scale - M_q ),
        * D_q is loaded from the same shape as M => [B, N, S],
        * 'q' indexes the Q dimension, 't' indexes K1 dimension, 'r' indexes K2 dimension.
    """

    block_idx = tl.program_id(0)  # block along K-dim in K1
    head_idx  = tl.program_id(1)  # which (batch, head)

    # Decompose head_idx => (batch_id, head_id)
    num_heads = NUM_HEADS
    batch_id = head_idx // num_heads
    head_id  = head_idx %  num_heads

    offset_q   = batch_id * q_strideB   + head_id * q_strideH
    offset_k1  = batch_id * k1_strideB  + head_id * k1_strideH
    offset_k2  = batch_id * k2_strideB  + head_id * k2_strideH
    offset_v1  = batch_id * v1_strideB  + head_id * v1_strideH
    offset_v2  = batch_id * v2_strideB  + head_id * v2_strideH
    offset_d   = batch_id * d_strideB   + head_id * d_strideH
    offset_dO  = batch_id * dO_strideB  + head_id * dO_strideH
    offset_dV1 = batch_id * dV1_strideB + head_id * dV1_strideH
    offset_dK1 = batch_id * dK1_strideB + head_id * dK1_strideH
    offset_m   = batch_id * M_strideB   + head_id * M_strideH

    start_kv = block_idx * BLOCK_KV
    offs_kv  = start_kv + tl.arange(0, BLOCK_KV)
    offs_dim = tl.arange(0, HEAD_DIM)

    # shape => [BLOCK_KV, HEAD_DIM] if we do (offs_kv[:,None], offs_dim[None,:])
    K1_block = tl.load(
        K1 + (offset_k1
              + offs_kv[:, None]*k1_strideS
              + offs_dim[None, :]*k1_strideD),
        mask = offs_kv[:, None] < SEQ_LEN,
        other=0.0
    ).to(tl.float32)

    # Load V1 block => shape [HEAD_DIM, BLOCK_KV]
    v1_block = tl.load(
        V1 + (offset_v1
              + offs_kv[None, :]*v1_strideS
              + offs_dim[:, None]*v1_strideD),
        mask = (offs_kv[None, :] < SEQ_LEN) & (offs_dim[:, None] < HEAD_DIM),
        other=0.0
    ).to(tl.float32)

    dV1_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK1_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    for block_q in range(0, SEQ_LEN, BLOCK_Q):
        offs_q = block_q + tl.arange(0, BLOCK_Q)

        # Q block => shape [BLOCK_Q, HEAD_DIM]
        Q_block = tl.load(
            Q + (offset_q
                 + offs_q[:, None]*q_strideS
                 + offs_dim[None, :]*q_strideD),
            mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)

        # dO block => shape [BLOCK_Q, HEAD_DIM]
        dO_block = tl.load(
            dO + (offset_dO
                  + offs_q[:, None]*dO_strideS
                  + offs_dim[None, :]*dO_strideD),
            mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)


        # dO * V1 => shape [BLOCK_Q, HEAD_KV]
        dO_v1_block = tl.dot(dO_block, v1_block)
        dO_v1_block = tl.trans(dO_v1_block, 1, 0)

        # M => shape [BLOCK_Q]; M is [B, N, S]
        m_q = tl.load(
            M + (offset_m + offs_q * M_strideS),
            mask = offs_q < SEQ_LEN,
            other=float('-inf')
        )

        # Precompute Q*K1 => shape: [BLOCK_Q, BLOCK_KV, HEAD_DIM]
        # QK1_block => (BLOCK_KV * BLOCK_Q, HEAD_DIM)
        QK1_block = Q_block[None, :, :] * K1_block[:, None, :]
        QK1_block = tl.reshape(QK1_block, (BLOCK_KV * BLOCK_Q, HEAD_DIM))

        # Loop over the K2 dimension in blocks
        for block_kv_2 in range(0, SEQ_LEN, BLOCK_KV):
            offs_kv2 = block_kv_2 + tl.arange(0, BLOCK_KV)

            # K2 => shape [HEAD_DIM, BLOCK_KV] 
            K2_block = tl.load(
                K2 + (offset_k2
                      + offs_kv2[None, :]*k2_strideS
                      + offs_dim[:, None]*k2_strideD),
                mask = (offs_kv2[None, :] < SEQ_LEN) & (offs_dim[:, None] < HEAD_DIM),
                other=0.0
            ).to(tl.float32)

            # Dot => shape: [ (BLOCK_KV * BLOCK_Q), BLOCK_KV ]
            QKK_block = tl.dot(QK1_block, K2_block)
            QKK_block = tl.reshape(QKK_block, (BLOCK_KV, BLOCK_Q, BLOCK_KV))

            P_block = tl.exp(QKK_block * softmax_scale - m_q[None, :, None])

            QKK_summed = tl.sum(P_block, axis=2)
            dV1_block += tl.dot(QKK_summed, dO_block)

            # V2 => shape [BLOCK_KV, HEAD_DIM] with the same logic
            v2_block = tl.load(
                V2 + (offset_v2
                      + offs_kv2[None, :]*v2_strideS
                      + offs_dim[:, None]*v2_strideD),
                mask = (offs_kv2[None, :] < SEQ_LEN),
                other=0.0
            ).to(tl.float32)

            dO_v2_block = tl.dot(dO_block, v2_block)
            dS_block = dO_v1_block[:, :, None] + dO_v2_block[None, :, :]

            # ---------------------------------------
            # D is [B, N, S], so we fetch D_q
            # ---------------------------------------
            D_block = tl.load(
                D + (offset_d + offs_q * d_strideS),
                mask = (offs_q < SEQ_LEN),
                other=0.0
            ).to(tl.float32)
            # shape => [BLOCK_Q], broadcast to [BLOCK_KV, BLOCK_Q, BLOCK_KV], (k1, q, k2)
            D_brd = D_block[None, :, None]  # => [1, BLOCK_Q, 1]

            # (dS - D) => shape [BLOCK_KV, BLOCK_Q, BLOCK_KV], (k1, q, k2)
            alpha = P_block * (dS_block - D_brd)
            alpha = softmax_scale*tl.reshape(alpha, (BLOCK_KV, BLOCK_Q*BLOCK_KV))
            

            K2_brd = tl.trans(K2_block, 1, 0)
            Q_brd  = Q_block[:, None, :]          # [BLOCK_Q, 1, HEAD_DIM]
            K2_brd = K2_brd[None, :, :]           # [1, BLOCK_KV, HEAD_DIM]
            sumQK2_block = Q_brd * K2_brd         # [BLOCK_Q, BLOCK_KV, HEAD_DIM]
            sumQK2_block = tl.reshape(sumQK2_block, (BLOCK_Q*BLOCK_KV, HEAD_DIM)) # [BLOCK_Q*BLOCK_KV, HEAD_DIM]
            
            dK1_sum = tl.dot(alpha, sumQK2_block) # Todo: Accumulate in tl dot

            dK1_block += dK1_sum

    tl.store(
        dV1 + (offset_dV1
               + offs_kv[:, None]*dV1_strideS
               + offs_dim[None, :]*dV1_strideD),
        dV1_block,
        mask = (offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
    )

    tl.store(
        dK1 + (offset_dK1
               + offs_kv[:, None]*dK1_strideS
               + offs_dim[None, :]*dK1_strideD),
        dK1_block,
        mask = (offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
    )