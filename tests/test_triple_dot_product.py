"""Correctness tests: triple_dot_product Triton kernels vs the PyTorch reference.

Requires a CUDA GPU (Triton); skipped otherwise.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is required for the kernels")
if not torch.cuda.is_available():
    pytest.skip("Triton kernels require a CUDA GPU", allow_module_level=True)

from flash_trittention.triple_dot_product.flash_trittention import flash_trittention
from flash_trittention.triple_dot_product.trittention_pytorch import (
    Trittention_pytorch,
)

torch.manual_seed(0)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)


def make_inputs(batch, heads, seq_len, head_dim, dtype, requires_grad=True):
    tensors = []
    for _ in range(5):
        t = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=dtype
        )
        t.requires_grad_(requires_grad)
        tensors.append(t)
    return tensors  # q, k1, k2, v1, v2


def reference_output(q, k1, k2, v1, v2, causal, softmax_scale, windows):
    k1_window, k2_window, kk_left, kk_right = windows
    ref = Trittention_pytorch(
        causal=causal,
        softmax_scale=softmax_scale,
        n_ctx=q.shape[2],
        k1_window=k1_window,
        k2_window=k2_window,
        kk_left=kk_left,
        kk_right=kk_right,
    )
    z, *_ = ref(q.float(), k1.float(), k2.float(), v1.float(), v2.float())
    return z


@cuda_required
@pytest.mark.parametrize("seq_len", [33, 64, 100])
@pytest.mark.parametrize("causal", [True, False])
def test_forward_and_backward_fp32(seq_len, causal):
    batch, heads, head_dim = 2, 2, 32
    softmax_scale = 1 / 8
    windows = (seq_len, seq_len, seq_len, seq_len)

    q, k1, k2, v1, v2 = make_inputs(batch, heads, seq_len, head_dim, torch.float32)
    q_r, k1_r, k2_r, v1_r, v2_r = (
        t.detach().clone().requires_grad_(True) for t in (q, k1, k2, v1, v2)
    )

    out = flash_trittention(
        q,
        k1,
        k2,
        v1,
        v2,
        softmax_scale=softmax_scale,
        k1_window=windows[0],
        k2_window=windows[1],
        kk_left=windows[2],
        kk_right=windows[3],
        causal=causal,
    )
    ref = reference_output(
        q_r, k1_r, k2_r, v1_r, v2_r, causal, softmax_scale, windows
    )

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)

    dO = torch.randn_like(out)
    out.backward(dO)
    ref.backward(dO)

    for name, a, b in (
        ("dq", q.grad, q_r.grad),
        ("dk1", k1.grad, k1_r.grad),
        ("dk2", k2.grad, k2_r.grad),
        ("dv1", v1.grad, v1_r.grad),
        ("dv2", v2.grad, v2_r.grad),
    ):
        torch.testing.assert_close(
            a, b, rtol=2e-2, atol=5e-3, msg=lambda m, n=name: f"{n}: {m}"
        )


@cuda_required
@pytest.mark.parametrize("seq_len", [64, 100])
def test_narrow_windows_causal(seq_len):
    batch, heads, head_dim = 2, 2, 32
    softmax_scale = 1 / 8
    windows = (8, 8, 4, 4)  # k1_window, k2_window, kk_left, kk_right

    q, k1, k2, v1, v2 = make_inputs(batch, heads, seq_len, head_dim, torch.float32)
    q_r, k1_r, k2_r, v1_r, v2_r = (
        t.detach().clone().requires_grad_(True) for t in (q, k1, k2, v1, v2)
    )

    out = flash_trittention(
        q,
        k1,
        k2,
        v1,
        v2,
        softmax_scale=softmax_scale,
        k1_window=windows[0],
        k2_window=windows[1],
        kk_left=windows[2],
        kk_right=windows[3],
        causal=True,
    )
    ref = reference_output(q_r, k1_r, k2_r, v1_r, v2_r, True, softmax_scale, windows)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)

    dO = torch.randn_like(out)
    out.backward(dO)
    ref.backward(dO)
    for name, a, b in (
        ("dq", q.grad, q_r.grad),
        ("dk1", k1.grad, k1_r.grad),
        ("dk2", k2.grad, k2_r.grad),
        ("dv1", v1.grad, v1_r.grad),
        ("dv2", v2.grad, v2_r.grad),
    ):
        torch.testing.assert_close(
            a, b, rtol=2e-2, atol=5e-3, msg=lambda m, n=name: f"{n}: {m}"
        )


@cuda_required
def test_forward_fp16_loose():
    batch, heads, seq_len, head_dim = 2, 2, 64, 32
    softmax_scale = 1 / 8

    q, k1, k2, v1, v2 = make_inputs(
        batch, heads, seq_len, head_dim, torch.float16, requires_grad=False
    )
    out = flash_trittention(
        q,
        k1,
        k2,
        v1,
        v2,
        softmax_scale=softmax_scale,
        k1_window=seq_len,
        k2_window=seq_len,
        kk_left=seq_len,
        kk_right=seq_len,
        causal=True,
    )
    ref = reference_output(
        q, k1, k2, v1, v2, True, softmax_scale, (seq_len,) * 4
    )
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2e-2)


@cuda_required
def test_rejects_mismatched_shapes():
    q, k1, k2, v1, v2 = make_inputs(2, 2, 32, 32, torch.float32)
    bad_v2 = v2[:, :, :16, :].contiguous()
    with pytest.raises(ValueError):
        flash_trittention(q, k1, k2, v1, bad_v2)


@cuda_required
def test_rejects_non_pow2_head_dim():
    q, k1, k2, v1, v2 = make_inputs(2, 2, 32, 24, torch.float32)
    with pytest.raises(ValueError):
        flash_trittention(q, k1, k2, v1, v2)
