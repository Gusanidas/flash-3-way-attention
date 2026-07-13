"""Correctness + performance harness for the B200 kernel vs baseline.

Run from repo root (broker run dir) with:
    PYTHONPATH=src:b200 python3 b200/bench_b200.py
Prints parseable CHECK/PERF lines.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch
from triton.testing import do_bench

from flash_trittention.triple_dot_product.flash_trittention import flash_trittention
from flash_trittention.triple_dot_product.trittention_pytorch import (
    Trittention_pytorch,
)
from b200.flash_trittention_b200 import flash_trittention_b200

DEVICE = "cuda"
SCALE = 1 / 8


def make(b, h, s, d, dtype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [
        torch.randn(b, h, s, d, device=DEVICE, dtype=dtype, generator=g)
        for _ in range(5)
    ]


def reference(q, k1, k2, v1, v2, causal, windows):
    k1w, k2w, kkl, kkr = windows
    ref = Trittention_pytorch(
        causal=causal,
        softmax_scale=SCALE,
        n_ctx=q.shape[2],
        k1_window=k1w,
        k2_window=k2w,
        kk_left=kkl,
        kk_right=kkr,
    )
    z, *_ = ref(q.float(), k1.float(), k2.float(), v1.float(), v2.float())
    return z


def check_correctness():
    ok = True
    for s, causal, windows_kind in [
        (64, True, "full"),
        (100, True, "full"),
        (100, False, "full"),
        (100, True, "narrow"),
        (33, True, "full"),
    ]:
        windows = (s, s, s, s) if windows_kind == "full" else (16, 16, 8, 8)
        inputs = make(2, 2, s, 32, torch.float32, seed=s)
        leaves = [t.clone().requires_grad_(True) for t in inputs]
        ref_leaves = [t.clone().requires_grad_(True) for t in inputs]

        out = flash_trittention_b200(
            *leaves,
            softmax_scale=SCALE,
            k1_window=windows[0],
            k2_window=windows[1],
            kk_left=windows[2],
            kk_right=windows[3],
            causal=causal,
        )
        ref = reference(*ref_leaves, causal, windows)
        fwd_err = (out - ref).abs().max().item()

        dO = torch.randn_like(out)
        out.backward(dO)
        ref.backward(dO)

        def rel_ok(a, b, atol, rtol=2e-2):
            return bool(((a - b).abs() <= atol + rtol * b.abs()).all().item())

        grad_errs = {}
        grads_pass = True
        for name, a, b in zip(
            ["dq", "dk1", "dk2", "dv1", "dv2"],
            [t.grad for t in leaves],
            [t.grad for t in ref_leaves],
        ):
            grad_errs[name] = (a - b).abs().max().item()
            grads_pass = grads_pass and rel_ok(a, b, atol=5e-3)

        # rtol+atol like torch.testing.assert_close / the pytest suite: both
        # triton kernels use TF32 dots, so absolute-only tolerances are too strict.
        passed = rel_ok(out, ref, atol=2e-3) and grads_pass
        ok = ok and passed
        print(
            f"CHECK s={s} causal={causal} win={windows_kind} "
            f"fwd_err={fwd_err:.2e} "
            + " ".join(f"{k}={v:.2e}" for k, v in grad_errs.items())
            + f" -> {'PASS' if passed else 'FAIL'}"
        )
    return ok


def bench_forward():
    cases = [
        # (S, causal, windows, label)
        (256, True, None, "full"),
        (512, True, None, "full"),
        (1024, True, None, "full"),
        (2048, True, (128, 128, 32, 32), "win128"),
    ]
    for s, causal, windows, label in cases:
        w = windows or (s, s, s, s)
        q, k1, k2, v1, v2 = make(2, 8, s, 64, torch.float16, seed=1)
        kwargs = dict(
            softmax_scale=SCALE,
            k1_window=w[0],
            k2_window=w[1],
            kk_left=w[2],
            kk_right=w[3],
            causal=causal,
        )
        with torch.no_grad():
            base_out = flash_trittention(q, k1, k2, v1, v2, **kwargs)
            b200_out = flash_trittention_b200(q, k1, k2, v1, v2, **kwargs)
            agree = (base_out - b200_out).abs().max().item()
            if agree > 5e-3:
                raise AssertionError(
                    f"B200 fp16 output diverged from baseline at s={s}: {agree:.3e}"
                )

            base_ms = do_bench(
                lambda: flash_trittention(q, k1, k2, v1, v2, **kwargs)
            )
            b200_ms = do_bench(
                lambda: flash_trittention_b200(q, k1, k2, v1, v2, **kwargs)
            )
        try:
            from b200.fwd_b200 import _tritt_fwd_b200

            best = _tritt_fwd_b200.best_config
        except Exception:
            best = "?"
        print(
            f"PERF s={s} {label} base_ms={base_ms:.3f} b200_ms={b200_ms:.3f} "
            f"speedup={base_ms / b200_ms:.2f}x agree={agree:.2e} cfg=[{best}]"
        )


if __name__ == "__main__":
    print(f"gpu={torch.cuda.get_device_name(0)}")
    ok = check_correctness()
    if not ok:
        print("RESULT correctness=FAIL (skipping perf)")
        sys.exit(1)
    bench_forward()
    print("RESULT correctness=PASS")
