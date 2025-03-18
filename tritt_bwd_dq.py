# @title Dq
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _tritt_bwd_dq(
    Q, K1, K2,
    V1, V2,
    D,
    softmax_scale,
    dO,
    dQ,             
    M,             
    # strides for Q
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
    # strides for dQ
    dQ_strideB, dQ_strideH, dQ_strideS, dQ_strideD,
    # strides for M
    M_strideB, M_strideH, M_strideS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr = False
):
    """
    Backward kernel to compute dQ.
    
    dQ = sum_{t,r} P_{q,t,r} * [(dO_q * V1_t + dO_q * V2_r) - D_q] * [K1_t * K2_r]
    Where:
        * P_{q,t,r} = exp(QK1K2 * softmax_scale - M_q)
        * D_q is loaded from the same shape as M => [B, N, S]
        * 'q' indexes the Q dimension, 't' indexes K1 dimension, 'r' indexes K2 dimension
    """
    
    block_idx = tl.program_id(0)  # block along Q-dim
    head_idx = tl.program_id(1)   # which (batch, head)
    
    num_heads = NUM_HEADS
    batch_id = head_idx // num_heads
    head_id = head_idx % num_heads
    
    offset_q = batch_id * q_strideB + head_id * q_strideH
    offset_k1 = batch_id * k1_strideB + head_id * k1_strideH
    offset_k2 = batch_id * k2_strideB + head_id * k2_strideH
    offset_v1 = batch_id * v1_strideB + head_id * v1_strideH
    offset_v2 = batch_id * v2_strideB + head_id * v2_strideH
    offset_d = batch_id * d_strideB + head_id * d_strideH
    offset_dO = batch_id * dO_strideB + head_id * dO_strideH
    offset_dQ = batch_id * dQ_strideB + head_id * dQ_strideH
    offset_m = batch_id * M_strideB + head_id * M_strideH
    
    start_q = block_idx * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    offs_dim = tl.arange(0, HEAD_DIM)
    
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    
    Q_block = tl.load(
        Q + (offset_q
             + offs_q[:, None] * q_strideS
             + offs_dim[None, :] * q_strideD),
        mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
        other=0.0
    ).to(tl.float32)
    
    # Load dO block => shape [BLOCK_Q, HEAD_DIM]
    dO_block = tl.load(
        dO + (offset_dO
              + offs_q[:, None] * dO_strideS
              + offs_dim[None, :] * dO_strideD),
        mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
        other=0.0
    ).to(tl.float32)
    
    # D => shape [BLOCK_Q]; D is [B, N, S]
    D_block = tl.load(
        D + (offset_d + offs_q * d_strideS),
        mask = offs_q < SEQ_LEN,
        other=0.0
    ).to(tl.float32)
    
    # M => shape [BLOCK_Q]; M is [B, N, S]
    m_q = tl.load(
        M + (offset_m + offs_q * M_strideS),
        mask = offs_q < SEQ_LEN,
        other=float('-inf')
    )
    
    # Loop over K1 in blocks
    for block_k1 in range(0, SEQ_LEN, BLOCK_KV):
        offs_k1 = block_k1 + tl.arange(0, BLOCK_KV)
        
        # Load K1 block => shape [BLOCK_KV, HEAD_DIM]
        K1_block = tl.load(
            K1 + (offset_k1
                 + offs_k1[:, None] * k1_strideS
                 + offs_dim[None, :] * k1_strideD),
            mask = (offs_k1[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)
        
        # Load V1 block => shape [BLOCK_KV, HEAD_DIM]
        V1_block = tl.load(
            V1 + (offset_v1
                 + offs_k1[None, :] * v1_strideS
                 + offs_dim[:, None] * v1_strideD),
            mask = (offs_k1[None, :] < SEQ_LEN) & (offs_dim[:, None] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)
        
        # Precompute dO * V1 => shape [BLOCK_Q, BLOCK_KV]
        dO_V1_block = tl.dot(dO_block, V1_block)

        QK1_block = Q_block[:, None, :] * K1_block[None,:, :]
        QK1_block = tl.reshape(QK1_block, (BLOCK_KV * BLOCK_Q, HEAD_DIM))
        
        # Loop over K2 dimension in blocks
        for block_k2 in range(0, SEQ_LEN, BLOCK_KV):
            offs_k2 = block_k2 + tl.arange(0, BLOCK_KV)
            
            # K2 block => shape [BLOCK_KV, HEAD_DIM]
            K2_block = tl.load(
                K2 + (offset_k2
                     + offs_k2[None, :] * k2_strideS
                     + offs_dim[:, None] * k2_strideD),
                mask = (offs_k2[None, :] < SEQ_LEN) & (offs_dim[:, None] < HEAD_DIM),
                other=0.0
            ).to(tl.float32)
            
            # V2 block => shape [BLOCK_KV, HEAD_DIM]
            V2_block = tl.load(
                V2 + (offset_v2
                     + offs_k2[None, :] * v2_strideS
                     + offs_dim[:, None] * v2_strideD),
                mask = (offs_k2[None, :] < SEQ_LEN) & (offs_dim[:, None] < HEAD_DIM),
                other=0.0
            ).to(tl.float32)
            
            
            # dO * V2 => shape [BLOCK_Q, BLOCK_KV]
            dO_V2_block = tl.dot(dO_block, V2_block)
            
            # Compute QK1K2 term => shape [BLOCK_Q, BLOCK_KV, BLOCK_KV]
            QKK_block = tl.dot(QK1_block, K2_block)
            QKK_block = tl.reshape(QKK_block, (BLOCK_Q, BLOCK_KV, BLOCK_KV))

            # P_block = exp( QKK_block * softmax_scale - m_q )
            P_block = tl.exp(QKK_block * softmax_scale - m_q[:, None, None])
            K2_block = tl.trans(K2_block, 1, 0)
            K1K2_block = K1_block[:, None, :] * K2_block[None, :, :]  # [BLOCK_KV, BLOCK_KV, HEAD_DIM]
            K1K2_block = tl.reshape(K1K2_block, (BLOCK_KV*BLOCK_KV, HEAD_DIM))
            
            
            dS_block = (dO_V1_block[:, :, None] + dO_V2_block[:, None, :]) - D_block[:, None, None]
            alpha = softmax_scale * P_block * dS_block
            alpha = tl.reshape(alpha, (BLOCK_Q, BLOCK_KV*BLOCK_KV)) # TODO: Probably too big, have to chunk

            dQ_block += tl.dot(alpha, K1K2_block) # TODO: Change to acumulate inside the dot
    
    # Store dQ
    tl.store(
        dQ + (offset_dQ
             + offs_q[:, None] * dQ_strideS
             + offs_dim[None, :] * dQ_strideD),
        dQ_block,
        mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < HEAD_DIM)
    )