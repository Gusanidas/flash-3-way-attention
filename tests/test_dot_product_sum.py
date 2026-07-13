"""Correctness tests: dot_product_sum Triton kernels vs the PyTorch reference.

Requires a CUDA GPU (Triton); skipped otherwise.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is required for the kernels")
if not torch.cuda.is_available():
    pytest.skip("Triton kernels require a CUDA GPU", allow_module_level=True)

from flash_trittention.dot_product_sum.trittention_triton import TrittentionTriton
from flash_trittention.dot_product_sum.trittention_pytorch import Trittention_pytorch

torch.manual_seed(0)


def make_inputs(batch, heads, seq_len, head_dim, dtype, requires_grad=True):
    tensors = []
    for _ in range(5):
        t = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=dtype)
        t.requires_grad_(requires_grad)
        tensors.append(t)
    return tensors  # q, k1, k2, v1, v2


def reference_output(q, k1, k2, v1, v2, causal, softmax_scale, k_diff):
    ref = Trittention_pytorch(
        causal=causal,
        softmax_scale=softmax_scale,
        n_ctx=q.shape[2],
        k_diff=k_diff,
    )
    z, *_ = ref(q.float(), k1.float(), k2.float(), v1.float(), v2.float())
    return z


def run_and_compare(seq_len, causal, k_diff, batch=2, heads=2, head_dim=32):
    softmax_scale = 1 / 8
    q, k1, k2, v1, v2 = make_inputs(batch, heads, seq_len, head_dim, torch.float32)
    q_r, k1_r, k2_r, v1_r, v2_r = (
        t.detach().clone().requires_grad_(True) for t in (q, k1, k2, v1, v2)
    )

    module = TrittentionTriton(
        causal=causal, softmax_scale=softmax_scale, k_diff=k_diff
    )
    out, _m = module(q, k1, k2, v1, v2)
    ref = reference_output(q_r, k1_r, k2_r, v1_r, v2_r, causal, softmax_scale, k_diff)

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


@pytest.mark.parametrize("seq_len", [33, 64, 100])
@pytest.mark.parametrize("causal", [True, False])
def test_forward_and_backward_fp32(seq_len, causal):
    # seq_len=33/100 exercise partial tail blocks; causal=True with seq_len>=32
    # exercises the off-diagonal/diagonal split that previously leaked future
    # positions when BLOCK_SIZE_Q was not a multiple of BLOCK_SIZE_KV.
    run_and_compare(seq_len, causal, k_diff=seq_len)


@pytest.mark.parametrize("k_diff", [2, 8])
def test_narrow_k_diff_causal(k_diff):
    run_and_compare(seq_len=64, causal=True, k_diff=k_diff)


def test_large_logits_no_overflow():
    # Large K1/K2 magnitudes previously overflowed the un-stabilized
    # exp(scale * K1.K2) accumulators in forward and bwd_dk1_dv1.
    # x8 makes scale*K1.K2 span several hundred — enough to overflow a raw
    # exp(scale*K1.K2) (the pre-fix code) while keeping the softmax margin large
    # against TF32 logit noise (x20 makes it one-hot enough that tiny numeric
    # differences flip the argmax and the value comparison becomes meaningless).
    seq_len, softmax_scale = 64, 1 / 8
    q, k1, k2, v1, v2 = make_inputs(2, 2, seq_len, 32, torch.float32)
    with torch.no_grad():
        k1.mul_(8.0)
        k2.mul_(8.0)
    q_r, k1_r, k2_r, v1_r, v2_r = (
        t.detach().clone().requires_grad_(True) for t in (q, k1, k2, v1, v2)
    )

    # ieee: at these magnitudes TF32 logit noise (~0.5 absolute) flips near-tied
    # softmax winners, which is not what this test is about — it guards overflow.
    module = TrittentionTriton(
        causal=True,
        softmax_scale=softmax_scale,
        k_diff=seq_len,
        input_precision="ieee",
    )
    out, _m = module(q, k1, k2, v1, v2)
    assert torch.isfinite(out).all(), "forward produced non-finite values"

    ref = reference_output(q_r, k1_r, k2_r, v1_r, v2_r, True, softmax_scale, seq_len)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)

    dO = torch.randn_like(out)
    out.backward(dO)
    for name, g in (("dq", q.grad), ("dk1", k1.grad), ("dk2", k2.grad),
                    ("dv1", v1.grad), ("dv2", v2.grad)):
        assert torch.isfinite(g).all(), f"{name} contains non-finite values"


def test_non_contiguous_inputs():
    # Forward previously applied Q's batch/head strides to all tensors without
    # calling .contiguous(), silently reading garbage for transposed inputs.
    seq_len = 64
    q, _, _, v1, v2 = make_inputs(2, 4, seq_len, 32, torch.float32)
    k1 = torch.randn(2, seq_len, 4, 32, device="cuda").transpose(1, 2)
    k2 = torch.randn(2, seq_len, 4, 32, device="cuda").transpose(1, 2)
    k1.requires_grad_(True)
    k2.requires_grad_(True)

    module = TrittentionTriton(causal=True, softmax_scale=1 / 8, k_diff=seq_len)
    out, _m = module(q, k1, k2, v1, v2)
    ref = reference_output(q, k1, k2, v1, v2, True, 1 / 8, seq_len)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)


def test_rejects_mismatched_shapes():
    q, k1, k2, v1, v2 = make_inputs(2, 2, 32, 32, torch.float32)
    bad_v2 = v2[:, :, :16, :].contiguous()
    module = TrittentionTriton()
    with pytest.raises(ValueError):
        module(q, k1, k2, v1, bad_v2)
