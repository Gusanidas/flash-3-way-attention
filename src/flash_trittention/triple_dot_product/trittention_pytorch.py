import einops
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from .kernels.utils import process_masking_variables


class Trittention_pytorch(nn.Module):

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: float = 1 / 8,
        n_ctx: int = 1024,
        k1_window: Optional[int] = None,
        k2_window: Optional[int] = None,
        kk_left: Optional[int] = None,
        kk_right: Optional[int] = None,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.softmax_scale = softmax_scale
        self.causal = causal
        k1_window, k2_window, kk_left, kk_right = process_masking_variables(
            n_ctx, k1_window, k2_window, kk_left, kk_right
        )
        self.k1_window = k1_window
        self.k2_window = k2_window
        self.kk_left = kk_left
        self.kk_right = kk_right

        self.precomputed_mask = self.create_causal_mask(
            n_ctx, k1_window, k2_window, kk_left, kk_right
        )

    def create_causal_mask(
        self,
        max_seq_len: int,
        k1_window: int,
        k2_window: int,
        kk_left: int,
        kk_right: int,
    ):

        j_indices = (
            torch.arange(max_seq_len)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # Shape: (1,1, t, 1, 1)
        k_indices = (
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
            (j_indices > q_indices)
            | (k_indices > q_indices)
            | (q_indices > j_indices + k1_window)
            | (q_indices > k_indices + k2_window)
            | (j_indices + kk_left < k_indices)
            | (k_indices + kk_right < j_indices)
        )
        return mask

    def forward(self, q, k1, k2, v1, v2):
        bs, ts, nh, d = q.shape

        attn_score = torch.einsum("b n j h, b n k h, b n q h -> b n j k q", k1, k2, q)

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

        # M = M + torch.log(L)
        v = F.silu(v1.unsqueeze(3)) * v2.unsqueeze(2)
        v_gated = v.clone()
        v = einops.rearrange(v, "b n p t h -> b n h (p t)")
        z = torch.einsum("bnql, bnhl -> bnqh", attn_score, v)
        return z, pre_softmax_attn_score, v_gated, M, L

    def apply_causal_mask(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        b, nn, tt, s, q = attn_scores.shape
        mask = self.precomputed_mask[:, :, :tt, :s, :q].to(attn_scores.device)
        attn_scores.masked_fill_(mask, -1e6)
        return attn_scores
