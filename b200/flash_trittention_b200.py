"""B200-tuned flash trittention (triple_dot_product variant).

Forward uses the restructured kernel in fwd_b200.py; backward reuses the
existing kernels from the main package (the saved M/L semantics are identical).
"""

import torch
import torch.nn.functional as F
import triton

from flash_trittention.triple_dot_product.flash_trittention import FlashTrittention
from flash_trittention.triple_dot_product.kernels.utils import (
    process_masking_variables,
)
from .fwd_b200 import _tritt_fwd_b200


class FlashTrittentionB200(FlashTrittention):
    @staticmethod
    def forward(
        ctx,
        Q,
        K1,
        K2,
        V1,
        V2,
        softmax_scale,
        kk_left,
        kk_right,
        k1_window,
        k2_window,
        causal,
        convert_to_float32,
    ):
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Narrow kk windows make the valid (j, k) region a thin band around the
        # diagonal; the baseline's small 16^3 tiles fit that band better than
        # this kernel's wide-BKV1 tiles (measured 0.88x). Dispatch to the
        # baseline forward there — backward is shared anyway.
        _k1w, _k2w, _kkl, _kkr = process_masking_variables(
            seq_len, k1_window, k2_window, kk_left, kk_right
        )
        if causal and (_kkl + _kkr) < seq_len // 8:
            return FlashTrittention.forward(
                ctx,
                Q,
                K1,
                K2,
                V1,
                V2,
                softmax_scale,
                kk_left,
                kk_right,
                k1_window,
                k2_window,
                causal,
                convert_to_float32,
            )

        for name, tensor in (("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)):
            if tensor.shape != Q.shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != Q shape")
            if tensor.dtype != Q.dtype or tensor.device != Q.device:
                raise ValueError(f"{name} dtype/device mismatch with Q")
        if head_dim != triton.next_power_of_2(head_dim):
            raise ValueError(f"head_dim must be a power of 2, got {head_dim}")

        # Preserve the raw input used by the inherited backward's SiLU
        # derivative.  Saving the caller's tensor directly would make backward
        # depend on any mutation performed after this forward call.
        V1_input = V1.clone()
        V1 = F.silu(V1)

        Q = Q.contiguous()
        K1 = K1.contiguous()
        K2 = K2.contiguous()
        V1 = V1.contiguous()
        V2 = V2.contiguous()

        k1_window, k2_window, kk_left, kk_right = process_masking_variables(
            seq_len, k1_window, k2_window, kk_left, kk_right
        )

        O = torch.zeros_like(Q)
        M = torch.zeros(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=Q.device
        )
        L = torch.zeros(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=Q.device
        )

        grid = lambda meta: (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
        )

        _tritt_fwd_b200[grid](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            softmax_scale=softmax_scale,
            M=M,
            L=L,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K1_seq=K1.stride(2),
            stride_K1_dim=K1.stride(3),
            stride_K2_seq=K2.stride(2),
            stride_K2_dim=K2.stride(3),
            stride_V1_seq=V1.stride(2),
            stride_V1_dim=V1.stride(3),
            stride_V2_seq=V2.stride(2),
            stride_V2_dim=V2.stride(3),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=causal,
            convert_to_float32=convert_to_float32,
        )

        M = M + torch.log(L)
        ctx.save_for_backward(Q, K1, K2, V2, O, M, V1_input)
        ctx.softmax_scale = softmax_scale
        ctx.kk_left = kk_left
        ctx.kk_right = kk_right
        ctx.k1_window = k1_window
        ctx.k2_window = k2_window
        ctx.causal = causal
        ctx.convert_to_float32 = convert_to_float32

        return O

    # backward inherited from FlashTrittention


def flash_trittention_b200(
    q,
    k1,
    k2,
    v1,
    v2,
    softmax_scale=1 / 8,
    kk_left=1,
    kk_right=1,
    k1_window=1,
    k2_window=1,
    causal=True,
    convert_to_float32=False,
):
    return FlashTrittentionB200.apply(
        q,
        k1,
        k2,
        v1,
        v2,
        softmax_scale,
        kk_left,
        kk_right,
        k1_window,
        k2_window,
        causal,
        convert_to_float32,
    )
