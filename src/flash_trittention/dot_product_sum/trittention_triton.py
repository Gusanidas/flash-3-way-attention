import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import triton
import triton.language as tl

# Import the kernels directly
from .kernels.fwd_kernel import _tritt_fwd
from .kernels.bwd_dq_kernel import _tritt_bwd_dq
from .kernels.bwd_dk1_dv1_kernel import _tritt_bwd_dk_dv
from .kernels.bwd_preprocess import _tritt_bwd_preprocess


class TrittentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k1,
        k2,
        v1,
        v2,
        causal,
        softmax_scale,
        k_diff,
        convert_to_float32=False,
        input_precision=None,
    ):
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(q, k1, k2, v1, v2)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.k_diff = k_diff
        ctx.convert_to_float32 = convert_to_float32
        ctx.input_precision = input_precision

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = q.shape
        o = torch.empty_like(q)
        m = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=q.device, dtype=torch.float32
        )

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        stride_Q_batch = q.stride(0)
        stride_Q_head = q.stride(1)
        stride_Q_seq = q.stride(2)
        stride_Q_dim = q.stride(3)

        stride_K1_seq = k1.stride(2)
        stride_K1_dim = k1.stride(3)

        stride_K2_seq = k2.stride(2)
        stride_K2_dim = k2.stride(3)

        stride_V1_seq = v1.stride(2)
        stride_V1_dim = v1.stride(3)

        stride_V2_seq = v2.stride(2)
        stride_V2_dim = v2.stride(3)

        stride_O_seq = o.stride(2)
        stride_O_dim = o.stride(3)

        # Use STAGE=3 for causal attention, STAGE=1 for non-causal attention
        STAGE = 3 if causal else 1

        _tritt_fwd[grid](
            Q=q,
            K1=k1,
            K2=k2,
            V1=v1,
            V2=v2,
            softmax_scale=softmax_scale,
            M=m,
            O=o,
            stride_Q_batch=stride_Q_batch,
            stride_Q_head=stride_Q_head,
            stride_Q_seq=stride_Q_seq,
            stride_Q_dim=stride_Q_dim,
            stride_K1_seq=stride_K1_seq,
            stride_K1_dim=stride_K1_dim,
            stride_K2_seq=stride_K2_seq,
            stride_K2_dim=stride_K2_dim,
            stride_V1_seq=stride_V1_seq,
            stride_V1_dim=stride_V1_dim,
            stride_V2_seq=stride_V2_seq,
            stride_V2_dim=stride_V2_dim,
            stride_O_seq=stride_O_seq,
            stride_O_dim=stride_O_dim,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=STAGE,
            k_diff=k_diff,
            convert_to_float32=convert_to_float32,
            input_precision=input_precision,
        )

        # Save additional tensors needed for backward
        ctx.save_for_backward(q, k1, k2, v1, v2, o, m)

        return o, m

    @staticmethod
    def backward(ctx, grad_output, m_grad=None):
        q, k1, k2, v1, v2, o, m = ctx.saved_tensors
        dtype = o.dtype

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = q.shape

        assert (
            grad_output.shape == o.shape
        ), f"Shape mismatch: O={o.shape}, dO={grad_output.shape}"
        assert o.device == grad_output.device, "O and dO must be on the same device"
        assert o.dtype == grad_output.dtype, "O and dO must have the same dtype"

        d = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=o.device, dtype=o.dtype
        )
        dq = torch.zeros_like(q)
        dk1 = torch.zeros_like(k1)
        dk2 = torch.zeros_like(k2).to(torch.float32)
        dv1 = torch.zeros_like(v1)
        dv2 = torch.zeros_like(v2).to(torch.float32)
        dk2_from_dk1_dv1 = torch.zeros_like(k2).to(torch.float32)

        q = q.contiguous()
        k1 = k1.contiguous()
        k2 = k2.contiguous()
        v1 = v1.contiguous()
        v2 = v2.contiguous()
        o = o.contiguous()
        grad_output = grad_output.contiguous()
        m = m.contiguous()
        v1_silu = F.silu(v1)

        q_strides = q.stride()
        k1_strides = k1.stride()
        k2_strides = k2.stride()
        v1_strides = v1.stride()
        v2_strides = v2.stride()
        o_strides = o.stride()
        grad_output_strides = grad_output.stride()
        m_strides = m.stride()
        d_strides = d.stride()
        dq_strides = dq.stride()
        dk1_strides = dk1.stride()
        dk2_strides = dk2.stride()
        dv1_strides = dv1.stride()
        dv2_strides = dv2.stride()

        BLOCK_SIZE_Q_PREPROCESS = 16
        grid_preprocess = (
            triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q_PREPROCESS),
            BATCH_SIZE * NUM_HEADS,
        )

        _tritt_bwd_preprocess[grid_preprocess](
            o,
            grad_output,
            d,
            grad_output_strides[2],
            grad_output_strides[3],
            o_strides[2],
            o_strides[3],
            SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q_PREPROCESS,
            HEAD_DIM=HEAD_DIM,
        )

        d = d.contiguous()
        d_strides = d.stride()

        grid_dq = lambda META: (
            triton.cdiv(SEQ_LEN, META["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )

        _tritt_bwd_dq[grid_dq](
            q,
            k1,
            k2,
            v1_silu,
            v2,
            d,
            ctx.softmax_scale,
            grad_output,
            dq,
            dk2,
            m,
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
            grad_output_strides[0],
            grad_output_strides[1],
            grad_output_strides[2],
            grad_output_strides[3],
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
            ctx.causal,
            ctx.k_diff,
            ctx.convert_to_float32,
            ctx.input_precision,
        )

        grid_dk1_dv1 = lambda META: (
            BATCH_SIZE * NUM_HEADS,
            triton.cdiv(SEQ_LEN, META["BLOCK_SIZE_KV"]),
        )

        _tritt_bwd_dk_dv[grid_dk1_dv1](
            q,
            k1,
            k2,
            v1_silu,
            v2,
            d,
            ctx.softmax_scale,
            grad_output,
            m,
            dk1,
            dv1,
            dv2,
            dk2_from_dk1_dv1,
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
            grad_output_strides[0],
            grad_output_strides[1],
            grad_output_strides[2],
            grad_output_strides[3],
            m_strides[0],
            m_strides[1],
            m_strides[2],
            dk1_strides[0],
            dk1_strides[1],
            dk1_strides[2],
            dk1_strides[3],
            dv1_strides[0],
            dv1_strides[1],
            dv1_strides[2],
            dv1_strides[3],
            dv2_strides[0],
            dv2_strides[1],
            dv2_strides[2],
            dv2_strides[3],
            dk2_strides[0],
            dk2_strides[1],
            dk2_strides[2],
            dk2_strides[3],
            SEQ_LEN,
            HEAD_DIM,
            NUM_HEADS,
            ctx.causal,
            ctx.k_diff,
            ctx.convert_to_float32,
            ctx.input_precision,
        )

        # Apply SiLU derivative to dV1
        sigmoid_v1 = torch.sigmoid(v1)
        silu_derivative = sigmoid_v1 * (1 + v1 * (1 - sigmoid_v1))
        dv1 = dv1 * silu_derivative

        # Combine dk2 from both processes
        dk2 = dk2 + dk2_from_dk1_dv1

        dk2 = dk2.to(dtype)
        dv2 = dv2.to(dtype)

        # === RETURN THE OUTPUT ===
        print(
            f"Return dq: {dq.shape}, dk1: {dk1.shape}, dk2: {dk2.shape}, dv1: {dv1.shape}, dv2: {dv2.shape}"
        )
        return dq, dk1, dk2, dv1, dv2, None, None, None, None, None


class TrittentionTriton(nn.Module):
    def __init__(
        self,
        causal: bool = True,
        softmax_scale: float = 1 / 8,
        k_diff: int = 1024,
        convert_to_float32: bool = False,
        input_precision: Optional[str] = None,
    ):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.k_diff = k_diff
        self.convert_to_float32 = convert_to_float32
        self.input_precision = input_precision

    def forward(self, q, k1, k2, v1, v2):
        """
        Forward pass for Trittention using Triton kernels.

        Args:
            q: Query tensor [B, N, S, D]
            k1: First key tensor [B, N, S, D]
            k2: Second key tensor [B, N, S, D]
            v1: First value tensor [B, N, S, D]
            v2: Second value tensor [B, N, S, D]

        Returns:
            Output tensor [B, N, S, D]
        """
        return TrittentionTritonFunction.apply(
            q,
            k1,
            k2,
            v1,
            v2,
            self.causal,
            self.softmax_scale,
            self.k_diff,
            self.convert_to_float32,
            self.input_precision,
        )
