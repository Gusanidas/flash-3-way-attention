"""Debug harness: localize the s=100 causal error in the b200 kernel and
probe the dot_product_sum causal NaN. Run: PYTHONPATH=src:b200 python3 b200/debug_b200.py [b200|dps]
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch

DEVICE = "cuda"
SCALE = 1 / 8


def make(b, h, s, d, dtype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [
        torch.randn(b, h, s, d, device=DEVICE, dtype=dtype, generator=g)
        for _ in range(5)
    ]


def debug_b200():
    from flash_trittention.triple_dot_product.flash_trittention import (
        flash_trittention,
    )
    from flash_trittention.triple_dot_product.trittention_pytorch import (
        Trittention_pytorch,
    )
    from b200.flash_trittention_b200 import flash_trittention_b200

    s = 100
    q, k1, k2, v1, v2 = make(2, 2, s, 32, torch.float32, seed=s)
    kwargs = dict(
        softmax_scale=SCALE,
        k1_window=s,
        k2_window=s,
        kk_left=s,
        kk_right=s,
        causal=True,
    )
    with torch.no_grad():
        base = flash_trittention(q, k1, k2, v1, v2, **kwargs)
        b200 = flash_trittention_b200(q, k1, k2, v1, v2, **kwargs)
    ref_mod = Trittention_pytorch(
        causal=True, softmax_scale=SCALE, n_ctx=s,
        k1_window=s, k2_window=s, kk_left=s, kk_right=s,
    )
    with torch.no_grad():
        ref, *_ = ref_mod(q, k1, k2, v1, v2)

    print(f"cfg={os.environ.get('B200_FORCE_CONFIG', 'autotune')}")
    print(f"max|base-ref|  = {(base - ref).abs().max().item():.3e}")
    print(f"max|b200-ref|  = {(b200 - ref).abs().max().item():.3e}")
    print(f"max|b200-base| = {(b200 - base).abs().max().item():.3e}")

    # per-q-row max error (b200 vs base), collapsed over batch/head/dim
    row_err = (b200 - base).abs().amax(dim=(0, 1, 3))
    bad = (row_err > 1e-3).nonzero().flatten().tolist()
    print(f"rows with |b200-base|>1e-3: {bad}")
    top = row_err.argsort(descending=True)[:8].tolist()
    print("worst rows:", [(r, f"{row_err[r].item():.2e}") for r in top])


def debug_dps():
    from flash_trittention.dot_product_sum.trittention_triton import TrittentionTriton
    from flash_trittention.dot_product_sum.trittention_pytorch import (
        Trittention_pytorch,
    )

    for s in [33, 64, 100]:
        q, k1, k2, v1, v2 = make(2, 2, s, 32, torch.float32, seed=s)
        module = TrittentionTriton(causal=True, softmax_scale=SCALE, k_diff=s)
        with torch.no_grad():
            out, m = module(q, k1, k2, v1, v2)
        ref_mod = Trittention_pytorch(causal=True, softmax_scale=SCALE, n_ctx=s, k_diff=s)
        with torch.no_grad():
            ref, _, _, ref_m = ref_mod(q, k1, k2, v1, v2)

        nan_rows = (
            (~torch.isfinite(out)).float().amax(dim=(0, 1, 3)).nonzero().flatten().tolist()
        )
        err = torch.where(torch.isfinite(out), (out - ref).abs(), torch.zeros_like(out))
        row_err = err.amax(dim=(0, 1, 3))
        bad = (row_err > 1e-3).nonzero().flatten().tolist()
        m_err = torch.where(
            torch.isfinite(m), (m - ref_m).abs(), torch.full_like(m, float("inf"))
        ).amax(dim=(0, 1))
        m_bad = (m_err > 1e-2).nonzero().flatten().tolist()
        print(f"s={s}: nan_rows={nan_rows[:20]}{'...' if len(nan_rows) > 20 else ''}")
        print(f"  rows |out-ref|>1e-3: {bad[:20]}{'...' if len(bad) > 20 else ''}")
        print(f"  rows |m-ref_m|>1e-2: {m_bad[:20]}{'...' if len(m_bad) > 20 else ''}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "b200"
    print(f"== {which} ==")
    if which == "b200":
        debug_b200()
    else:
        debug_dps()
