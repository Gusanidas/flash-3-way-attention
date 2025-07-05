import torch
import torch.nn.functional as F


def slow_trittention_dot_product_sum(
    q, k1, k2, v1, v2, causal_mask=False, softmax_scale=1.0, k_diff=4
):
    """
    Slow trittention implementation with sum of dot products attention scoring.

    Key differences from original:
    1. Attention score = dot(k1,k2) + dot(q,k2) (sum of two dot products)
    2. Value computed with SwishGLU: silu(v1) * v2

    Args:
        q, k1, k2, v1, v2: tensors of shape (Batch, n_heads, seq_len, d_head)
        causal_mask: bool, whether to apply causal masking
        softmax_scale: float, scaling factor for attention scores

    Returns:
        output: tensor of shape (Batch, n_heads, seq_len, d_head)
    """
    bs, n_heads, seq_len, d_head = q.shape
    out = torch.zeros((bs, n_heads, seq_len, d_head), device=q.device)
    pre_softmax_attn_score = torch.zeros(
        (bs, n_heads, seq_len, seq_len, seq_len), device=q.device
    )
    v_gated = torch.zeros((bs, n_heads, seq_len, seq_len, d_head), device=q.device)
    for bi in range(bs):
        for hi in range(n_heads):
            for qi in range(seq_len):
                q_vec = q[bi, hi, qi, :]

                scores = []
                swishglu_values = []

                for si in range(seq_len):  # k1 position
                    for ti in range(seq_len):  # k2 position
                        if causal_mask and (si > ti or ti > qi or ti - si > k_diff):
                            continue

                        k1_vec = k1[bi, hi, si, :]
                        k2_vec = k2[bi, hi, ti, :]
                        v1_vec = v1[bi, hi, si, :]
                        v2_vec = v2[bi, hi, ti, :]

                        # Sum of dot products: dot(k1,k2) + dot(q,k2)
                        score_k1k2 = torch.dot(k1_vec, k2_vec)
                        score_qk2 = torch.dot(q_vec, k2_vec)
                        pre_softmax_attn_score[bi, hi, qi, si, ti] = (
                            score_k1k2 + score_qk2
                        )
                        total_score = score_k1k2 + score_qk2

                        # SwishGLU value: silu(v1) * v2
                        swishglu_val = F.silu(v1_vec) * v2_vec
                        v_gated[bi, hi, si, ti, :] = swishglu_val

                        scores.append(total_score.item() * softmax_scale)
                        swishglu_values.append(swishglu_val)

                if len(scores) > 0:
                    scores_tensor = torch.tensor(scores, device=q.device)
                    attn_weights = F.softmax(scores_tensor, dim=0)

                    head_output = torch.zeros(d_head, device=q.device)
                    for idx, weight in enumerate(attn_weights):
                        head_output += weight * swishglu_values[idx]

                    out[bi, hi, qi, :] = head_output
                else:
                    print(f"No scores found for qi: {qi}")

    return out, pre_softmax_attn_score, v_gated
