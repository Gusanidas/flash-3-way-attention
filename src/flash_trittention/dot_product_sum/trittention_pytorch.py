import einops
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class Trittention_pytorch(nn.Module):

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: float = 1 / 8,
        n_ctx: int = 1024,
        k_diff: int = 1024,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.precomputed_mask = self.create_causal_mask(n_ctx, k_diff)
        self.k_diff = k_diff

    def create_causal_mask(self, max_seq_len: int, k_diff: int):

        t_indices = (
            torch.arange(max_seq_len)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # Shape: (1,1, t, 1, 1)
        s_indices = (
            torch.arange(max_seq_len)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(-1)
        )  # Shape: (1,1, 1, s, 1)
        q_indices = (
            torch.arange(max_seq_len)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # Shape: (1,1, 1, 1, q)
        mask = (
            (t_indices > s_indices)
            | (s_indices > q_indices)
            | (s_indices - t_indices > k_diff)
        )
        return mask

    def forward(self, q, k1, k2, v1, v2):

        attn_score_qk = torch.einsum("bnth, bnqh-> bntq", k2, q)
        attn_score_kk = torch.einsum("bnsh, bnth -> bnst", k1, k2)
        attn_score = attn_score_qk.unsqueeze(2) + attn_score_kk.unsqueeze(-1)

        if self.causal:
            attn_score = self.apply_causal_mask(attn_score)
        pre_softmax_attn_score = attn_score
        attn_score = (
            einops.rearrange(attn_score, "b n s t q -> b n q (s t)")
            * self.softmax_scale
        )
        pre_softmax_attn_score2 = attn_score.clone()

        M = attn_score.max(dim=-1, keepdim=False)[0]
        attn_score = attn_score - M.unsqueeze(-1)
        L = (torch.exp(attn_score)).sum(dim=-1, keepdim=False) + 0.01
        attn_score = attn_score.softmax(dim=-1)

        M = M + torch.log(L)
        sft2 = pre_softmax_attn_score2 - M.unsqueeze(-1)
        sft2 = torch.exp(sft2)
        sft2 = torch.sum(sft2, dim=-1, keepdim=False)
        v = F.silu(v1.unsqueeze(3)) * v2.unsqueeze(2)
        v_gated = v.clone()
        v = einops.rearrange(v, "b n p t h -> b n h (p t)")
        z = torch.einsum("bnql, bnhl -> bnqh", attn_score, v)
        return z, pre_softmax_attn_score, v_gated, M

    def apply_causal_mask(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        b, nn, tt, s, q = attn_scores.shape
        mask = self.precomputed_mask[:, :, :tt, :s, :q].to(attn_scores.device)
        attn_scores.masked_fill_(mask, -1e6)
        return attn_scores
