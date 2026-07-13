# H100 (Hopper, sm_90a) forward kernel for triple_dot_product

A Hopper-targeted forward kernel for the triple-dot-product trittention.
Backward reuses the main package's kernels unchanged (the saved `M`/`L`
semantics are identical).

## Usage

```python
from h100.flash_trittention_h100 import flash_trittention_h100
out = flash_trittention_h100(q, k1, k2, v1, v2, softmax_scale=1/8, causal=True,
                             k1_window=None, k2_window=None,
                             kk_left=None, kk_right=None,
                             use_tma=True)   # or False for plain loads
```

Same signature/semantics as `flash_trittention` from the main package, plus
`use_tma` (default: env `H100_USE_TMA`, off unless set to `1`).

## What's in it

Shares the structural wins of the B200 kernel (see `b200/README.md`): the
`QK2[(q,k),d] = Q·K2` precompute that turns the inner loop into one dense
tensor-core matmul per iteration, fp32/TF32 score formation, base-2
exponentials with the scale folded into Q, the interior-tile fast path that
skips mask construction, and the narrow-window dispatch to the baseline. On
top of that, two Hopper-specific pieces:

1. **TMA descriptor loads** (`use_tma=True`). K1/K2/V1/V2 tiles are loaded
   through device-side tensor descriptors (`tl.make_tensor_descriptor`),
   which lower to the sm_90 TMA engine: tile address math leaves the SM,
   loads issue as bulk async copies, and out-of-bounds rows are zero-filled
   by hardware instead of by mask construction. Requires contiguous inputs
   (the wrapper enforces this) and a host-side scratch allocator
   (`triton.set_allocator`, installed automatically).

2. **Hopper-shaped autotune space** keyed on `(HEAD_DIM, SEQ_LEN, USE_TMA)`
   (the two load paths tune independently). The winning shapes keep the main
   matmul's row count `BQ*BKV2` at 64–128 — one wgmma M-tile per warpgroup
   with `num_warps=4` — and lean on `num_stages=3` pipelining, vs the B200
   winner's 256 rows / 8 warps / 2 stages.

## Measured — H100 SXM 80GB (torch 2.9.1+cu130, triton 3.5.1; B=2, H=8, D=64, fp16)

`H100_FORCE_CONFIG=bq,bkv2,bkv1,warps,stages` pins a config for additional
sweeps on other software stacks.

| case | baseline | h100 plain | h100 TMA |
|---|---|---|---|
| S=256, causal, full windows | 0.634 ms | 0.298 ms (2.13x) | 0.337 ms (1.88x) |
| S=512, causal, full windows | 3.562 ms | 1.451 ms (2.45x) | 1.830 ms (1.95x) |
| S=1024, causal, full windows | 23.29 ms | 8.748 ms (2.66x) | 11.18 ms (2.08x) |
| S=2048, causal, windows 128/kk 32 | 1.163 ms | dispatched | dispatched |

(Winning configs used BQ=8, BKV2=16, BKV1=64, and 4 warps, with 3–4 stages.)

On this H100/Triton stack, plain loads were 12–23% faster than TMA, so the
wrapper defaults to the plain path while retaining TMA as an explicit option.

**Narrow-window dispatch:** as in the b200 wrapper, `causal and
kk_left + kk_right < seq_len // 8` falls back to the baseline forward, whose
small 16^3 tiles fit the thin diagonal (j, k) band better. Correctness of this
kernel on narrow windows is still covered by the CHECK cases (s=100 stays
above the dispatch threshold).

## Files

- `fwd_h100.py` — the Triton kernel (both load paths) + autotune configs.
- `flash_trittention_h100.py` — autograd wrapper; backward = main package.
- `bench_h100.py` — correctness (vs the PyTorch reference, fwd + all 5 grads,
  both load paths) and forward benchmarks (baseline vs plain vs TMA).
