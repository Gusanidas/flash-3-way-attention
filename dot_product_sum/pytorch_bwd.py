import torch

# from bwd_preprocess import tritt_bwd_preprocess, pytorch_bwd_preprocess
import einops
import torch.nn.functional as F


def pytorch_bwd_preprocess(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of the backward preprocessing.

    Computes D = sum(O * dO) along the head dimension for each sequence position.
    This is equivalent to the Triton kernel but uses PyTorch operations.

    Args:
        O: Output tensor from forward pass, shape [B, H, S, D]
        dO: Gradient of output tensor, shape [B, H, S, D]

    Returns:
        D: Preprocessing result, shape [B, H, S]
    """
    # Validate input shapes
    assert dO.shape == O.shape, f"Shape mismatch: O={O.shape}, dO={dO.shape}"
    assert O.device == dO.device, "O and dO must be on the same device"
    assert O.dtype == dO.dtype, "O and dO must have the same dtype"

    # Compute D = sum(O * dO) along the head dimension (dim=-1)
    D = torch.sum(O * dO, dim=-1)

    return D


def backward_v_operation(dv, V1, V2):
    """
    dv: gradient w.r.t. v, shape (b, n, p, t, h)
    V1: original V1, shape (b, n, p, h)
    V2: original V2, shape (b, n, t, h)

    Returns:
    dL_dV1: gradient w.r.t. V1, shape (b, n, p, h)
    dL_dV2: gradient w.r.t. V2, shape (b, n, t, h)
    """

    # Compute SiLU and its derivative
    V1_unsqueezed = V1.unsqueeze(3)  # (b, n, p, 1, h)
    V2_unsqueezed = V2.unsqueeze(2)  # (b, n, 1, t, h)

    silu_V1 = F.silu(V1_unsqueezed)  # (b, n, p, 1, h)

    # SiLU derivative: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_V1 = torch.sigmoid(V1)
    silu_derivative = sigmoid_V1 * (1 + V1 * (1 - sigmoid_V1))  # (b, n, p, h)

    # Gradient w.r.t. V2: sum over p dimension
    dL_dV2 = (dv * silu_V1).sum(dim=2)  # (b, n, t, h)

    # Gradient w.r.t. V1: sum over t dimension
    dL_dV1 = (dv * V2_unsqueezed * silu_derivative.unsqueeze(3)).sum(
        dim=3
    )  # (b, n, p, h)

    return dL_dV1, dL_dV2


def create_causal_mask(max_seq_len: int, k_diff: int):

    t_indices = (
        torch.arange(max_seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    )  # Shape: (1,1, t, 1, 1)
    s_indices = (
        torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
    )  # Shape: (1,1, 1, s, 1)
    q_indices = (
        torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )  # Shape: (1,1, 1, 1, q)
    mask = (
        (t_indices > s_indices)
        | (s_indices > q_indices)
        | (s_indices - t_indices > k_diff)
    )
    return mask


def pytorch_bwd(
    Q: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    V1: torch.Tensor,
    V2: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    softmax_scale: float,
    M: torch.Tensor,
    causal: bool = True,
    k_diff: int = 1024,
):
    # D = tritt_bwd_preprocess(O, dO)
    batch_size, n_heads, seq_len, head_dim = O.shape
    if causal:
        causal_mask = create_causal_mask(seq_len, k_diff)
        causal_mask = causal_mask.to(O.device)
    D = pytorch_bwd_preprocess(O, dO)
    print(f"Shape of D: {D.shape}")
    attn_score_qk = torch.einsum("bnth, bnqh-> bntq", K2, Q)
    attn_score_kk = torch.einsum("bnsh, bnth -> bnst", K1, K2)
    attn_score = attn_score_qk.unsqueeze(2) + attn_score_kk.unsqueeze(-1)
    pre_softmax_attn_score = attn_score.clone()
    print(f"Shape of pre_softmax_attn_score: {pre_softmax_attn_score.shape}")
    print(
        f"Avg of pre_softmax_attn_score: {pre_softmax_attn_score.mean().item():.6f}, Avg of abs of pre_softmax_attn_score: {pre_softmax_attn_score.abs().mean().item():.6f}"
    )
    print("............................................")
    if causal:
        attn_score.masked_fill_(causal_mask, -1e6)

    attn_score = (
        einops.rearrange(attn_score, "b n s t q -> b n q (s t)") * softmax_scale
    )
    attn_score = attn_score - M.unsqueeze(-1)
    # attn_score = F.softmax(attn_score, dim=-1)
    attn_score = torch.exp(attn_score)
    sums_attn_score = attn_score.sum(dim=-1, keepdim=False)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(
        f"Avg of sums_attn_score: {sums_attn_score.mean().item():.6f}, avg abs of sums_attn_score: {sums_attn_score.abs().mean().item():.6f}"
    )
    P = einops.rearrange(attn_score, "b n q (s t) -> b n q s t", s=seq_len)
    v = F.silu(V1.unsqueeze(3)) * V2.unsqueeze(2)
    print(f"Shape of dO: {dO.shape}")
    print(f"Shape of v: {v.shape}")
    dP = torch.einsum("b n q h, b n j k h -> b n q j k", dO, v)
    # dP = einops.rearrange(dP, "b n q (j k) -> b n q j k", j=O.shape[0])

    dV = torch.einsum("b n q j k, b n q h -> b n j k h", P, dO)
    dV1, dV2 = backward_v_operation(dV, V1, V2)
    dS = P * (dP - D.unsqueeze(-1).unsqueeze(-1))

    # Backpropagate through softmax_scale and rearrange
    # First undo the softmax scaling
    d_attn_score_rearranged = dS

    # Undo the rearrange from "b n q (s t)" back to "b n s t q"
    # d_attn_score = einops.rearrange(
    #    d_attn_score_rearranged, "b n q (s t) -> b n s t q", s=seq_len
    # )
    d_attn_score = d_attn_score_rearranged

    # Backpropagate through the addition:
    # attn_score = attn_score_qk.unsqueeze(2) + attn_score_kk.unsqueeze(-1)
    # For attn_score_qk: sum over the s dimension
    d_attn_score_qk = d_attn_score.sum(dim=3)  # [b, n, t, q]

    # For attn_score_kk: sum over the q dimension
    d_attn_score_kk = d_attn_score.sum(dim=2)  # [b, n, s, t]

    # Backpropagate through attn_score_qk = torch.einsum("bnth, bnqh-> bntq", K2, Q)
    dQ = torch.einsum("bnqt, bnth -> bnqh", d_attn_score_qk, K2) * softmax_scale

    # Backpropagate through attn_score_kk = torch.einsum("bnsh, bnth -> bnst", K1, K2)
    dK1 = torch.einsum("bnst, bnth -> bnsh", d_attn_score_kk, K2) * softmax_scale

    # K2 gets gradients from both paths
    dK2_from_qk = torch.einsum("bnqt, bnqh -> bnth", d_attn_score_qk, Q) * softmax_scale
    dK2_from_kk = (
        torch.einsum("bnst, bnsh -> bnth", d_attn_score_kk, K1) * softmax_scale
    )
    dK2 = dK2_from_qk + dK2_from_kk

    return dQ, dK1, dK2, dV1, dV2, dS


if __name__ == "__main__":
    # Create dummy tensors for testing
    batch_size = 2
    n_heads = 4
    seq_len = 8
    head_dim = 64

    # Initialize tensors with random values
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    K1 = torch.randn(batch_size, n_heads, seq_len, head_dim)
    K2 = torch.randn(batch_size, n_heads, seq_len, head_dim)
    V1 = torch.randn(batch_size, n_heads, seq_len, head_dim)
    V2 = torch.randn(batch_size, n_heads, seq_len, head_dim)
    O = torch.randn(batch_size, n_heads, seq_len, head_dim)
    dO = torch.randn(batch_size, n_heads, seq_len, head_dim)
    M = torch.randn(batch_size, n_heads, seq_len)
    softmax_scale = 0.125  # Typical value for attention scaling

    print("Testing pytorch_bwd function...")
    print(f"Input tensor shapes:")
    print(f"Q: {Q.shape}")
    print(f"K1: {K1.shape}")
    print(f"K2: {K2.shape}")
    print(f"V1: {V1.shape}")
    print(f"V2: {V2.shape}")
    print(f"O: {O.shape}")
    print(f"dO: {dO.shape}")
    print(f"softmax_scale: {softmax_scale}")
    print()

    dQ, dK1, dK2, *_ = pytorch_bwd(Q, K1, K2, V1, V2, O, dO, softmax_scale, M)
    print(f"Shape of dQ: {dQ.shape}")
    print(f"Shape of dK1: {dK1.shape}")
    print(f"Shape of dK2: {dK2.shape}")
