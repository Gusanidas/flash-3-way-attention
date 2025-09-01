#!/usr/bin/env python3
import time
from typing import Tuple

import torch

from flash_trittention.triple_dot_product.flash_trittention import flash_trittention
from flash_trittention.triple_dot_product.trittention_pytorch import Trittention_pytorch


# All relevant variables defined together (keep simple and explicit)
BATCH_SIZE = 2
NUM_HEADS = 2
SEQ_LEN = 444
HEAD_DIM = 64
DEVICE = "cuda"
DTYPE = torch.float32
DTYPE = torch.bfloat16

CAUSAL = True
SOFTMAX_SCALE = 1.0 / (HEAD_DIM**0.5)
K1_WINDOW = 40
K2_WINDOW = 40
KK_LEFT = None
KK_RIGHT = None

NUM_WARMUP = 5
NUM_RUNS = 20


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_tensors(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    Create input tensors and cloned copies for PyTorch and Flash Trittention.

    Returns two tuples, each containing: (q, k1, k2, v1, v2, dO)
    The PyTorch and Flash tensors contain the same values but have independent grads.
    """
    torch.manual_seed(42)
    base_q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    base_k1 = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    base_k2 = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    base_v1 = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    base_v2 = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    base_dO = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )

    q_pt = base_q.clone().detach().requires_grad_(True)
    k1_pt = base_k1.clone().detach().requires_grad_(True)
    k2_pt = base_k2.clone().detach().requires_grad_(True)
    v1_pt = base_v1.clone().detach().requires_grad_(True)
    v2_pt = base_v2.clone().detach().requires_grad_(True)
    dO_pt = base_dO.clone().detach()

    q_fl = base_q.clone().detach().requires_grad_(True)
    k1_fl = base_k1.clone().detach().requires_grad_(True)
    k2_fl = base_k2.clone().detach().requires_grad_(True)
    v1_fl = base_v1.clone().detach().requires_grad_(True)
    v2_fl = base_v2.clone().detach().requires_grad_(True)
    dO_fl = base_dO.clone().detach()

    return (q_pt, k1_pt, k2_pt, v1_pt, v2_pt, dO_pt), (
        q_fl,
        k1_fl,
        k2_fl,
        v1_fl,
        v2_fl,
        dO_fl,
    )


def time_forward(model_call, q, k1, k2, v1, v2) -> float:
    # Warmup
    for _ in range(NUM_WARMUP):
        out = model_call(q, k1, k2, v1, v2)
        if isinstance(out, tuple):
            out = out[0]
    _sync_cuda()

    # Timed
    times_ms = []
    for _ in range(NUM_RUNS):
        _sync_cuda()
        start = time.perf_counter()
        out = model_call(q, k1, k2, v1, v2)
        if isinstance(out, tuple):
            out = out[0]
        _sync_cuda()
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)
    return float(sum(times_ms) / len(times_ms))


def time_backward(model_call, q, k1, k2, v1, v2, dO) -> float:
    # Warmup
    for _ in range(NUM_WARMUP):
        out = model_call(q, k1, k2, v1, v2)
        if isinstance(out, tuple):
            out = out[0]
        out.backward(dO, retain_graph=True)
        for tensor in (q, k1, k2, v1, v2):
            if tensor.grad is not None:
                tensor.grad.zero_()
    _sync_cuda()

    # Timed (measure backward only)
    times_ms = []
    for _ in range(NUM_RUNS):
        out = model_call(q, k1, k2, v1, v2)
        if isinstance(out, tuple):
            out = out[0]
        _sync_cuda()
        start = time.perf_counter()
        out.backward(dO, retain_graph=True)
        _sync_cuda()
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)
        for tensor in (q, k1, k2, v1, v2):
            if tensor.grad is not None:
                tensor.grad.zero_()
    return float(sum(times_ms) / len(times_ms))


def main():
    print("Simple Trittention Benchmark\n" + "=" * 32)

    # Instantiate PyTorch reference and define Flash function wrapper
    trittention_pt = Trittention_pytorch(
        causal=CAUSAL,
        softmax_scale=SOFTMAX_SCALE,
        n_ctx=SEQ_LEN,
        k1_window=K1_WINDOW,
        k2_window=K2_WINDOW,
        kk_left=KK_LEFT,
        kk_right=KK_RIGHT,
    ).to(device=DEVICE, dtype=DTYPE)

    flash_call = lambda q, k1, k2, v1, v2: flash_trittention(
        q,
        k1,
        k2,
        v1,
        v2,
        softmax_scale=SOFTMAX_SCALE,
        kk_left=KK_LEFT,
        kk_right=KK_RIGHT,
        k1_window=K1_WINDOW,
        k2_window=K2_WINDOW,
        causal=CAUSAL,
        convert_to_float32=True,
        input_precision=None,
    )

    # Create inputs for both methods with identical values
    (q_pt, k1_pt, k2_pt, v1_pt, v2_pt, dO_pt), (
        q_fl,
        k1_fl,
        k2_fl,
        v1_fl,
        v2_fl,
        dO_fl,
    ) = create_tensors(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, DEVICE, DTYPE)

    # Forward benchmarks
    print("Forward (ms):")
    try:
        fwd_pt = time_forward(trittention_pt, q_pt, k1_pt, k2_pt, v1_pt, v2_pt)
        print(f"  PyTorch: {fwd_pt:.3f}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("  PyTorch: n/a")
        else:
            raise

    fwd_fl = time_forward(flash_call, q_fl, k1_fl, k2_fl, v1_fl, v2_fl)
    print(f"  Flash:   {fwd_fl:.3f}")

    # Backward benchmarks
    print("Backward (ms):")
    try:
        bwd_pt = time_backward(trittention_pt, q_pt, k1_pt, k2_pt, v1_pt, v2_pt, dO_pt)
        print(f"  PyTorch: {bwd_pt:.3f}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("  PyTorch: n/a")
        else:
            raise

    bwd_fl = time_backward(flash_call, q_fl, k1_fl, k2_fl, v1_fl, v2_fl, dO_fl)
    print(f"  Flash:   {bwd_fl:.3f}")


if __name__ == "__main__":
    main()
