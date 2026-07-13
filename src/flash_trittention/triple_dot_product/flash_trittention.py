import warnings
import torch
import torch.nn.functional as F
from .kernels.fwd import _tritt_fwd
from .kernels.bwd_dk1 import _tritt_bwd_dk1
from .kernels.bwd_dk2 import _tritt_bwd_dk2
from .kernels.bwd_dv1 import _tritt_bwd_dv1
from .kernels.bwd_dv2 import _tritt_bwd_dv2
from .kernels.bwd_dq import _tritt_bwd_dq
from .kernels.utils import process_masking_variables
import triton


class FlashTrittention(torch.autograd.Function):
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

        # Validate inputs: all tensors must share Q's shape, dtype and device.
        for name, tensor in (("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)):
            if tensor.shape != Q.shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match Q shape "
                    f"{tuple(Q.shape)}"
                )
            if tensor.dtype != Q.dtype:
                raise ValueError(
                    f"{name} dtype {tensor.dtype} does not match Q dtype {Q.dtype}"
                )
            if tensor.device != Q.device:
                raise ValueError(
                    f"{name} device {tensor.device} does not match Q device {Q.device}"
                )

        # The kernels index K1/K2/V1/V2/O with tl.arange(0, HEAD_DIM), which Triton
        # only accepts for power-of-2 lengths, so reject non-power-of-2 head_dim up
        # front with a clear message rather than an opaque Triton compile error.
        HEAD_DIM_pow2 = triton.next_power_of_2(head_dim)
        if head_dim != HEAD_DIM_pow2:
            raise ValueError(
                f"head_dim must be a power of 2, got {head_dim} "
                f"(next power of 2 is {HEAD_DIM_pow2})"
            )

        # Clone the raw V1 input (before silu) for the backward silu-derivative so
        # a later in-place mutation of the caller's V1 cannot corrupt the gradient.
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
        O = O.contiguous()
        M = M.contiguous()
        L = L.contiguous()

        grid = lambda meta: (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
        )

        sqrt_softmax_scale = softmax_scale**0.5

        _tritt_fwd[grid](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            softmax_scale=sqrt_softmax_scale,
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
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=causal,
            convert_to_float32=convert_to_float32,
        )

        M = M + torch.log(L)
        # Save tensors for backward
        ctx.save_for_backward(Q, K1, K2, V2, O, M, V1_input)
        ctx.softmax_scale = softmax_scale
        ctx.kk_left = kk_left
        ctx.kk_right = kk_right
        ctx.k1_window = k1_window
        ctx.k2_window = k2_window
        ctx.causal = causal
        ctx.convert_to_float32 = convert_to_float32

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K1, K2, V2, O, M, original_V1 = ctx.saved_tensors
        batch_size, num_heads, seq_len, head_dim = Q.shape
        HEAD_DIM_pow2 = triton.next_power_of_2(head_dim)

        # Common preprocessing for all backward kernels
        # Make tensors contiguous
        Q = Q.contiguous()
        K1 = K1.contiguous()
        K2 = K2.contiguous()
        V2 = V2.contiguous()
        dO = dO.contiguous()
        O = O.contiguous()
        M = M.contiguous()

        # Store original V1 before applying silu
        original_V1 = original_V1.contiguous()
        V1 = F.silu(original_V1)
        V1 = V1.contiguous()

        k1_window, k2_window, kk_left, kk_right = process_masking_variables(
            seq_len, ctx.k1_window, ctx.k2_window, ctx.kk_left, ctx.kk_right
        )

        # Compute D = sum(dO * O, dim=-1) in float32 to match the float32 M tensor.
        D = (dO.float() * O.float()).sum(-1)

        dq = torch.zeros_like(Q)
        dk1 = torch.zeros_like(K1)
        dk2 = torch.zeros_like(K2)
        dv1 = torch.zeros_like(V1)
        dv2 = torch.zeros_like(V2)

        convert_to_float32 = ctx.convert_to_float32

        sqrt_softmax_scale = ctx.softmax_scale**0.5
        masking_value = 1e6 if Q.dtype == torch.float32 else 1e4

        grid_q = lambda meta: (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
        )

        grid_kv = lambda meta: (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_KV"]),
            batch_size * num_heads,
        )

        # Call backward kernels directly
        _tritt_bwd_dq[grid_q](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            dO=dO,
            D=D,
            M=M,
            dq=dq,
            softmax_scale=sqrt_softmax_scale,
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
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            stride_dq_seq=dq.stride(2),
            stride_dq_dim=dq.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=ctx.causal,
            convert_to_float32=convert_to_float32,
            masking_value=masking_value,
        )

        _tritt_bwd_dk1[grid_kv](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            dO=dO,
            D=D,
            M=M,
            dk1=dk1,
            softmax_scale=ctx.softmax_scale,
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
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            stride_dk1_seq=dk1.stride(2),
            stride_dk1_dim=dk1.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=ctx.causal,
            convert_to_float32=convert_to_float32,
            masking_value=masking_value,
        )

        _tritt_bwd_dk2[grid_kv](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            dO=dO,
            D=D,
            M=M,
            dk2=dk2,
            softmax_scale=ctx.softmax_scale,
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
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            stride_dk2_seq=dk2.stride(2),
            stride_dk2_dim=dk2.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=ctx.causal,
            convert_to_float32=convert_to_float32,
            masking_value=masking_value,
        )

        _tritt_bwd_dv1[grid_kv](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            dO=dO,
            M=M,
            dv1=dv1,
            softmax_scale=ctx.softmax_scale,
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
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            stride_dv1_seq=dv1.stride(2),
            stride_dv1_dim=dv1.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=ctx.causal,
            convert_to_float32=convert_to_float32,
            masking_value=masking_value,
        )

        _tritt_bwd_dv2[grid_kv](
            Q=Q,
            K1=K1,
            K2=K2,
            V1=V1,
            V2=V2,
            dO=dO,
            M=M,
            dv2=dv2,
            softmax_scale=ctx.softmax_scale,
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
            stride_dO_seq=dO.stride(2),
            stride_dO_dim=dO.stride(3),
            stride_dv2_seq=dv2.stride(2),
            stride_dv2_dim=dv2.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            HEAD_DIM_pow2=HEAD_DIM_pow2,
            kk_left=kk_left,
            kk_right=kk_right,
            k1_window=k1_window,
            k2_window=k2_window,
            causal=ctx.causal,
            convert_to_float32=convert_to_float32,
            masking_value=masking_value,
        )

        # Apply silu derivative to dv1
        v1_sigmoid = torch.sigmoid(original_V1)
        dv1 = dv1 * v1_sigmoid * (1 + original_V1 * (1 - v1_sigmoid))

        return dq, dk1, dk2, dv1, dv2, None, None, None, None, None, None, None


def flash_trittention(
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
    input_precision=None,
):
    """
    Flash Trittention triton implementation

    Note: ``input_precision`` is accepted for backward compatibility with
    existing callers but is currently unused. Passing a non-default value has
    no effect and emits a warning so the ignored request is not silent.
    """
    if input_precision is not None:
        warnings.warn(
            "flash_trittention: `input_precision` is currently unused and has no "
            f"effect (got {input_precision!r}); matmuls run at the kernel's "
            "default precision.",
            stacklevel=2,
        )
    return FlashTrittention.apply(
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
