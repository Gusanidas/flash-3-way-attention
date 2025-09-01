from typing import Optional
import triton
import triton.language as tl


def process_masking_variables(
    seq_len: int,
    k1_window: Optional[int] = None,
    k2_window: Optional[int] = None,
    kk_left: Optional[int] = None,
    kk_right: Optional[int] = None,
):
    k1_window = k1_window if k1_window is not None else seq_len
    k2_window = k2_window if k2_window is not None else seq_len
    kk_left = kk_left if kk_left is not None else seq_len
    kk_right = kk_right if kk_right is not None else seq_len
    return k1_window, k2_window, kk_left, kk_right


# Configuration constants
SHORT_CONFIG = True  # Set to True for faster development, False for production
INVALID_LOGIT = -float("inf")
DEFAULT_MAX_LOGIT = -1000.0


def valid_config(bq, bkv, ns, nw):
    if bq > 32 and bkv > 32:
        return False
    if ns > 2 and (bq > 32 or bkv > 32):
        return False
    if ns > 1 and (bq > 64 or bkv > 64):
        return False
    if nw > 2 and (bq > 64 or bkv > 64):
        return False
    if nw > 4 and (bq > 32 or bkv > 32):
        return False
    return True


def get_autotune_config(short=SHORT_CONFIG):
    """Generate autotune configurations for Triton kernels."""
    if short:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=3, num_warps=8
            )
            for bq in [16]
            for bkv in [16]
        ]
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw
            )
            for bq in [16, 32]
            for bkv in [16, 32]
            for ns in [1, 2]
            for nw in [1, 4]
            if valid_config(bq, bkv, ns, nw)
        ]


@triton.jit
def get_causal_mask(
    kk_left,
    kk_right,
    k1_window,
    k2_window,
    offs_q,
    offs_kv1,
    offs_kv2,
):
    """Compute combined causal mask for Q-K1-K2 attention."""
    # Query-Key1 mask: Q can attend to K1 within window
    qk1_mask = (offs_q[:, None] <= offs_kv1[None, :] + k1_window) & (
        offs_q[:, None] >= offs_kv1[None, :]
    )

    # Query-Key2 mask: Q can attend to K2 within window
    qk2_mask = (offs_q[:, None] <= offs_kv2[None, :] + k2_window) & (
        offs_q[:, None] >= offs_kv2[None, :]
    )

    # Key1-Key2 interaction constraint
    kk_mask = (offs_kv1[None, :] + kk_left >= offs_kv2[:, None]) & (
        offs_kv2[:, None] + kk_right >= offs_kv1[None, :]
    )

    # Combined mask: (Q, K2, K1)
    return qk1_mask[:, None, :] & qk2_mask[:, :, None] & kk_mask[None, :, :]
