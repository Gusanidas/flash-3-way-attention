import triton
import triton.language as tl


@triton.jit
def intervals_overlap(
    start1: tl.int32, end1: tl.int32, start2: tl.int32, end2: tl.int32
) -> tl.int1:
    """Check if two intervals overlap."""
    return tl.maximum(start1, start2) < tl.minimum(end1, end2)


@triton.jit
def align_to_block(value: tl.int32, block_size: tl.int32) -> tl.int32:
    """
    Get the highest multiple of block_size that is <= value.
    For example: align_to_block(50, 16) = 48 (which is 3 * 16)
    """
    return (value // block_size) * block_size


@triton.jit
def interval_k2_from_q(
    causal: tl.int1,
    seq_len: tl.int32,
    start_q: tl.int32,
    block_q: tl.int32,
    block_k: tl.int32,
    k2_window: tl.int32,
) -> tl.int32:
    """
    Calculate the interval for k2.
    Returns start and end as tuple.
    """
    target = start_q - k2_window
    aligned_start = align_to_block(target, block_k)
    causal_start = tl.maximum(0, aligned_start)

    start = tl.where(causal, causal_start, 0)
    end = tl.where(causal, start_q + block_q, seq_len)

    return start, end


@triton.jit
def interval_k2_from_k1(
    causal: tl.int1,
    seq_len: tl.int32,
    start_k1: tl.int32,
    block_k: tl.int32,
    k1_window: tl.int32,
    kk_left: tl.int32,
    kk_right: tl.int32,
) -> tl.int32:
    if not causal:
        return 0, seq_len

    if k1_window is None:
        k1_window = seq_len

    if kk_left is None:
        kk_left = seq_len
    if kk_right is None:
        kk_right = seq_len

    start = start_k1 - kk_right
    start = align_to_block(start, block_k)
    start = tl.maximum(start, 0)
    end = start_k1 + block_k + kk_left
    end_k1 = start_k1 + block_k + k1_window
    end = tl.minimum(end, end_k1)
    end = tl.minimum(end, seq_len)
    return start, end


@triton.jit
def interval_k1_from_k2(
    causal: tl.int1,
    seq_len: tl.int32,
    start_k2: tl.int32,
    block_k: tl.int32,
    k2_window: tl.int32,
    kk_left: tl.int32,
    kk_right: tl.int32,
) -> tl.int32:
    if not causal:
        return 0, seq_len

    if k2_window is None:
        k2_window = seq_len
    if kk_left is None:
        kk_left = seq_len
    if kk_right is None:
        kk_right = seq_len

    start = start_k2 - kk_left
    start = align_to_block(start, block_k)
    start = tl.maximum(start, 0)
    end = start_k2 + block_k + kk_right
    end_k2 = start_k2 + block_k + k2_window
    end = tl.minimum(end, end_k2)

    return start, end


@triton.jit
def interval_k1(
    causal: tl.int1,
    seq_len: tl.int32,
    start_q: tl.int32,
    block_q: tl.int32,
    start_k2: tl.int32,
    block_k: tl.int32,
    k1_window: tl.int32,
    kk_left: tl.int32,
    kk_right: tl.int32,
) -> tl.int32:
    """
    Calculate the interval for k1 following the strictest constraints.
    Returns start and end as tuple.
    Note: Use negative values to indicate "None" constraints.
    """
    k1_target = start_q - k1_window
    k1_aligned = align_to_block(k1_target, block_k)
    k1_start_constraint = tl.maximum(0, k1_aligned)

    # kk_left constraint (use if kk_left >= 0)
    kk_target = start_k2 - kk_left
    kk_aligned = align_to_block(kk_target, block_k)
    kk_start_constraint = tl.maximum(0, kk_aligned)

    causal_start = tl.maximum(k1_start_constraint, kk_start_constraint)

    # Calculate end position
    causal_end = start_q + block_q

    # Apply kk_right constraint if active (>= 0)
    kk_end_constraint = start_k2 + block_k + kk_right
    causal_end = tl.minimum(causal_end, kk_end_constraint)

    start = tl.where(causal, causal_start, 0)
    end = tl.where(causal, causal_end, seq_len)

    return start, end


@triton.jit
def intervals_q(
    causal: tl.int1,
    seq_len: tl.int32,
    start_k1: tl.int32,
    block_k: tl.int32,
    start_k2: tl.int32,
    k1_window: tl.int32,
    k2_window: tl.int32,
    kk_left: tl.int32,
    kk_right: tl.int32,
) -> tl.int32:
    """
    Calculate the interval for q given k1 and k2 intervals.
    Returns start and end as tuple.
    """
    if not causal:
        return 0, seq_len

    # Lower bound is the max of start_k1 and start_k2
    start = tl.maximum(start_k1, start_k2)

    # Upper bound is the max of:
    # - start_k1 + block_k + k1_window
    # - start_k2 + block_k + k2_window
    end1 = start_k1 + block_k + k1_window
    end2 = start_k2 + block_k + k2_window
    end = tl.minimum(end1, end2)
    end = tl.minimum(end, seq_len)

    return start, end
