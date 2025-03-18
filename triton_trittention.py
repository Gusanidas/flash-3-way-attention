import torch
import triton
import triton.language as tl
from torch import nn
from tritt_fwd import _tritt_fwd
from tritt_bwd_dq import _tritt_bwd_dq
from tritt_bwd_dk_dv import _tritt_bwd_dv1_dk1
from bwd_preprocess import _tritt_bwd_preprocess

class TrittentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K1, K2, V1, V2, softmax_scale, causal):
        """
        Q, K1, K2, V1, V2: [B, H, S, D]
        causal: bool
        softmax_scale: float
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Output and auxiliary
        O = torch.empty_like(Q)  # [B, N, S, D]
        M = torch.empty((batch_size, num_heads, seq_len), device=Q.device, dtype=torch.float32)

        # Kernel grid definitions
        BLOCK_SIZE_Q = 16 # TODO: Add AutoTune
        BLOCK_SIZE_KV = 16

        # Strides
        stride_Q_batch = Q.stride(0) # TODO: Only one set of strides
        stride_Q_head  = Q.stride(1)
        stride_Q_seq   = Q.stride(2)
        stride_Q_dim   = Q.stride(3)

        stride_K1_seq  = K1.stride(2)
        stride_K1_dim  = K1.stride(3)
        stride_K2_seq  = K2.stride(2)
        stride_K2_dim  = K2.stride(3)

        stride_V1_seq  = V1.stride(2)
        stride_V1_dim  = V1.stride(3)
        stride_V2_seq  = V2.stride(2)
        stride_V2_dim  = V2.stride(3)

        stride_O_seq   = O.stride(2)
        stride_O_dim   = O.stride(3)

        STAGE = 3 if causal else 1

        # 2D grid over (S // BLOCK_SIZE_Q) x (B*N)
        grid = lambda meta: (triton.cdiv(seq_len, BLOCK_SIZE_Q), batch_size*num_heads, 1)

        # Launch forward kernel
        _tritt_fwd[grid](
            Q, K1, K2, V1, V2,
            softmax_scale,
            M,  # writes log-max per query
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
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            num_heads,
            seq_len,
            head_dim,
            STAGE
        )

        # Save for backward
        ctx.save_for_backward(Q, K1, K2, V1, V2, O, M)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.batch_size = batch_size
        ctx.num_heads = num_heads
        ctx.seq_len = seq_len
        ctx.head_dim = head_dim
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_KV = BLOCK_SIZE_KV

        return O, M

    @staticmethod
    def backward(ctx, dO, dM):
        """
        dO: [B, N, S, D]
        dM: [B, N, S] (the gradient wrt the log-sumexp "max" if used in computation).
            Often you won't backprop through M, but let's include it to be safe.
        """
        Q, K1, K2, V1, V2, O, M = ctx.saved_tensors
        batch_size, num_heads, seq_len, head_dim = ctx.batch_size, ctx.num_heads, ctx.seq_len, ctx.head_dim

        assert dO.is_contiguous()
        assert O.is_contiguous()
        assert dM.is_contiguous()
        assert Q.is_contiguous()
        assert K1.is_contiguous()
        assert K2.is_contiguous()
        assert V1.is_contiguous()
        assert V2.is_contiguous()
        assert Q.stride() == K1.stride() == V1.stride() == O.stride() == dO.stride() == K2.stride() == V2.stride()

        dQ  = torch.zeros_like(Q)
        dK1 = torch.zeros_like(K1)
        dK2 = torch.zeros_like(K2)
        dV1 = torch.zeros_like(V1)
        dV2 = torch.zeros_like(V2)

        dQ  = dQ.contiguous()
        dK1 = dK1.contiguous()
        dK2 = dK2.contiguous()
        dV1 = dV1.contiguous()
        dV2 = dV2.contiguous()

        # We also need a "D" buffer for the log-sumexp correction
        D = torch.zeros_like(M)  # shape [batch_size, num_heads, seq_len]

        q_strideB, q_strideH, q_strideS, q_strideD = Q.stride()
        k1_strideB, k1_strideH, k1_strideS, k1_strideD = K1.stride()
        k2_strideB, k2_strideH, k2_strideS, k2_strideD = K2.stride()
        v1_strideB, v1_strideH, v1_strideS, v1_strideD = V1.stride()
        v2_strideB, v2_strideH, v2_strideS, v2_strideD = V2.stride()

        dO_strideB, dO_strideH, dO_strideS, dO_strideD = dO.stride()
        dQ_strideB, dQ_strideH, dQ_strideS, dQ_strideD = dQ.stride()
        dV1_strideB, dV1_strideH, dV1_strideS, dV1_strideD = dV1.stride()
        dK1_strideB, dK1_strideH, dK1_strideS, dK1_strideD = dK1.stride()

        dV2_strideB, dV2_strideH, dV2_strideS, dV2_strideD = dV2.stride()
        M_strideB, M_strideH, M_strideS = M.stride()
        d_strideB, d_strideH, d_strideS = D.stride()

        # 2D grid
        grid = (seq_len//ctx.BLOCK_SIZE_Q, batch_size*num_heads)

        # 1) Preprocess: fill D
        _tritt_bwd_preprocess[grid](
            O, dO,
            D,
            dO_strideS, dO_strideD,
            O.stride(2), O.stride(3),
            seq_len,
            BLOCK_SIZE_Q=ctx.BLOCK_SIZE_Q,
            HEAD_DIM=head_dim
        )
        assert q_strideB == k1_strideB == k2_strideB == v1_strideB == v2_strideB == dO_strideB == dQ_strideB == dK1_strideB
        assert q_strideH == k1_strideH == k2_strideH == v1_strideH == v2_strideH == dO_strideH == dQ_strideH == dK1_strideH
        assert q_strideS == k1_strideS == k2_strideS == v1_strideS == v2_strideS == dO_strideS == dQ_strideS == dK1_strideS
        assert q_strideD == k1_strideD == k2_strideD == v1_strideD == v2_strideD == dO_strideD == dQ_strideD == dK1_strideD

        # 2) dV1, dK1
        _tritt_bwd_dv1_dk1[grid](
            Q, K1, K2,
            V1, V2,
            D,
            ctx.softmax_scale,
            dO,
            dV1,
            dK1,
            M,
            q_strideB, q_strideH, q_strideS, q_strideD,
            k1_strideB, k1_strideH, k1_strideS, k1_strideD,
            k2_strideB, k2_strideH, k2_strideS, k2_strideD,
            v1_strideB, v1_strideH, v1_strideS, v1_strideD,
            v2_strideB, v2_strideH, v2_strideS, v2_strideD,
            d_strideB, d_strideH, d_strideS,
            dO_strideB, dO_strideH, dO_strideS, dO_strideD,
            dV1_strideB, dV1_strideH, dV1_strideS, dV1_strideD,
            dK1_strideB, dK1_strideH, dK1_strideS, dK1_strideD,
            M_strideB, M_strideH, M_strideS,
            seq_len,
            BLOCK_Q=ctx.BLOCK_SIZE_Q,
            BLOCK_KV=ctx.BLOCK_SIZE_KV,
            HEAD_DIM=head_dim,
            NUM_HEADS=num_heads
        )

        # 3) dV2, dK2 (just reuse the same kernel, swapping K1->K2, V1->V2, etc.) TODO: Change when adding causal
        _tritt_bwd_dv1_dk1[grid](
            Q, K2, K1,
            V2, V1,
            D,
            ctx.softmax_scale,
            dO,
            dV2,
            # store into dK2
            dK2,
            M,
            # Q strides
            q_strideB, q_strideH, q_strideS, q_strideD,
            # K1=K2 strides
            k2_strideB, k2_strideH, k2_strideS, k2_strideD,
            # K2=K1 strides
            k1_strideB, k1_strideH, k1_strideS, k1_strideD,
            # V1=V2 strides
            v2_strideB, v2_strideH, v2_strideS, v2_strideD,
            # V2=V1 strides
            v1_strideB, v1_strideH, v1_strideS, v1_strideD,
            d_strideB, d_strideH, d_strideS,
            dO_strideB, dO_strideH, dO_strideS, dO_strideD,
            dV2_strideB, dV2_strideH, dV2_strideS, dV2_strideD,
            dK1_strideB, dK1_strideH, dK1_strideS, dK1_strideD,  # but actually for dK2
            M_strideB, M_strideH, M_strideS,
            seq_len,
            BLOCK_Q=ctx.BLOCK_SIZE_Q,
            BLOCK_KV=ctx.BLOCK_SIZE_KV,
            HEAD_DIM=head_dim,
            NUM_HEADS=num_heads
        )
##
        # 4) dQ
        _tritt_bwd_dq[grid](
            Q, K1, K2, V1, V2,
            D,
            ctx.softmax_scale,
            dO,
            dQ,
            M,
            q_strideB, q_strideH, q_strideS, q_strideD,
            k1_strideB, k1_strideH, k1_strideS, k1_strideD,
            k2_strideB, k2_strideH, k2_strideS, k2_strideD,
            v1_strideB, v1_strideH, v1_strideS, v1_strideD,
            v2_strideB, v2_strideH, v2_strideS, v2_strideD,
            d_strideB, d_strideH, d_strideS,
            dO_strideB, dO_strideH, dO_strideS, dO_strideD,
            dQ_strideB, dQ_strideH, dQ_strideS, dQ_strideD,
            M_strideB, M_strideH, M_strideS,
            seq_len,
            BLOCK_Q=ctx.BLOCK_SIZE_Q,
            BLOCK_KV=ctx.BLOCK_SIZE_KV,
            HEAD_DIM=head_dim,
            NUM_HEADS=num_heads
        )

        # Return grads in order of forward args:
        # Q, K1, K2, V1, V2, softmax_scale, causal
        return dQ, dK1, dK2, dV1, dV2, None, None


class TritonTrittention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = dropout
        # You can add linear layers etc. if needed

    def forward(self, q, k1, k2, v1, v2, softmax_scale=1.0, causal=False):
        # q, k1, k2, v1, v2: all expected shape [batch, heads, seq_len, dim_head]
        # returns O, M
        O, M = TrittentionFunction.apply(q, k1, k2, v1, v2, softmax_scale, causal)
        return O, M