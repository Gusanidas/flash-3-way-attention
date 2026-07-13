# B200 (Blackwell, sm_100a) forward kernel for triple_dot_product

A restructured forward kernel for the triple-dot-product trittention, tuned on
an NVIDIA B200. Backward reuses the main package's kernels unchanged (the saved
`M`/`L` semantics are identical).

## Usage

```python
from b200.flash_trittention_b200 import flash_trittention_b200
out = flash_trittention_b200(q, k1, k2, v1, v2, softmax_scale=1/8, causal=True,
                             k1_window=None, k2_window=None,
                             kk_left=None, kk_right=None)
```

Same signature/semantics as `flash_trittention` from the main package.

## What changed vs the baseline kernel

1. **No per-pair outer products.** The baseline materializes
   `k1[j,:] * K2[k,:]` as a `(D, BKV*BKV)` tensor inside the innermost loop —
   once per (kv1, kv2) block pair. Since
   `S[q,(j,k)] = sum_d (Q[q,d] * K2[k,d]) * K1[j,d]`, this kernel instead
   precomputes `QK2[(q,k), d] = Q[q,d] * K2[k,d]` once per kv2 block; each inner
   iteration is then a single dense matmul `(BQ*BKV2, D) @ (D, BKV1)` — tensor-
   core shaped, and `BKV1` can grow independently (the winning config uses 64
   vs the baseline's 16). The value side is handled the same way:
   `T = exp_P @ silu(V1)` (a matmul), then one multiply-reduce against `V2`.

2. **Base-2 exponentials.** `softmax_scale * log2(e)` is folded into Q once and
   the online softmax runs on `exp2` (the native SFU op); the stored `M` is
   converted back to ln units, so backward sees the standard convention.

3. **Interior-tile fast path.** Whole-tile validity is checked with a handful
   of scalar (non-divergent) comparisons; fully-valid tiles skip the 3-D mask
   construction and `tl.where` entirely. With full windows + causal, most tiles
   are interior, so this removes most of the masking cost.

4. **FP32 score formation.** The restructured `Q*K2` pre-product and its
   reduction against K1 use fp32/TF32 even when values remain fp16/bf16. This
   avoids an extra rounding step that caused visible D=64 forward and backward
   drift while retaining low-precision value accumulation.

5. **Blackwell-scaled autotune space** (10 configs), keyed on
   `(HEAD_DIM, SEQ_LEN)`. `BLOCK_SIZE_KV2` does not participate in the matmul
   shapes (rows are `BQ*BKV2`), so it can shrink below 16 to trade for a bigger
   `BQ`.

## Measured (B200 183GB, torch 2.9.1+cu130, triton 3.5.1; B=2, H=8, D=64, fp16)

| case | baseline | b200 | speedup |
|---|---|---|---|
| S=256, causal, full windows | 0.595 ms | 0.461 ms | 1.29x |
| S=512, causal, full windows | 3.380 ms | 2.111 ms | 1.60x |
| S=1024, causal, full windows | 21.50 ms | 11.61 ms | 1.85x |
| S=2048, causal, windows 128/kk 32 | 1.075 ms | 1.074 ms | 1.00x (dispatched to baseline) |

(Winning configs used BQ=16, BKV2=16, 8 warps, and 3 stages; BKV1 was 64 at
S=256/512 and 128 at S=1024.)

**Narrow-window dispatch:** with narrow kk windows the valid (j, k) region is a
thin band around the diagonal, so wide `BKV1` tiles are mostly masked and the
fast path rarely triggers — the baseline's small 16^3 tiles fit the band
better (measured 0.88x). The wrapper therefore dispatches to the baseline
forward when `causal and kk_left + kk_right < seq_len // 8`, making the b200
path never-worse. Tiling (kv1, kv2) jointly along the band diagonal is the
proper fix (future work).

Cost model note: per valid `(q, j, k)` element the two matmuls contribute ~256
flops on tensor cores while the softmax machinery (exp, mask, reductions,
rescaling) contributes ~8 CUDA-core/SFU ops — at B200 throughput ratios the
elementwise side dominates. That is why the wins come from exp2 + mask-skipping
rather than from the matmul restructure alone (which mostly *enables* the
bigger tiles).

## Files

- `fwd_b200.py` — the Triton kernel + autotune configs
  (`B200_FORCE_CONFIG=bq,bkv2,bkv1,warps` env var pins a single config).
- `flash_trittention_b200.py` — autograd wrapper; backward = main package.
- `bench_b200.py` — correctness (vs the PyTorch reference, fwd + all 5 grads)
  and forward benchmarks (vs the baseline Triton kernel).
- `debug_b200.py` — row-level error localization helpers used during bring-up.
