
import torch
import torch.nn as nn
import einops
from slow_trittention import slow_tri
from flash_trittention import tritt_fwd


class Trittention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.dim_head = dim_head
        self.heads = heads

    def forward(self, q, k1, k2, v1, v2, softmax_scale=1.0):
        bs, n_heads, seq_len, d_head = q.shape
        a, b, c, d, e = k1, k2, q, v1, v2

        attn_score = torch.einsum("bnsh, bnth, bnqh -> bnstq", a,b,c)
        attn_score = einops.rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")*softmax_scale
        max, _ = torch.max(attn_score, dim=-1)
        suma = torch.exp(attn_score - max).sum(dim=-1)
        attn_score = attn_score.softmax(dim=-1)
        attn_score = einops.rearrange(attn_score, "b n p1 (p2 p3) -> b n p1 p2 p3", p2 = seq_len, p3 = seq_len) 
        z = torch.einsum('bnqlr, bnld -> bnqd', attn_score, d) + torch.einsum('bnqlr, bnrd -> bnqd', attn_score, e)
        max = max + torch.log(suma)
        return z, max


if __name__ == "__main__":
    q = torch.randn(1, 8, 16, 64)
    k1 = torch.randn(1, 8, 16, 64)
    k2 = torch.randn(1, 8, 16, 64)
    v1 = torch.randn(1, 8, 16, 64)
    v2 = torch.randn(1, 8, 16, 64)
    tritt = Trittention(64, 8, 64)
    tritt_out = tritt(q, k1, k2, v1, v2)
    tritt_out_flash, _ = tritt_fwd(q, k1, k2, v1, v2, causal=True, softmax_scale=1.0)
    slow_out = slow_tri(q, k1, k2, v1, v2)

    print(tritt_out.shape)
    print(tritt_out_flash.shape)
    print(slow_out.shape)

    print(f"Slow and tritt")
    print(f"max diff: {torch.max(tritt_out - slow_out)}")
    print(f"min diff: {torch.min(tritt_out - slow_out)}")
    print(f"mean abs diff: {torch.mean(torch.abs(tritt_out - slow_out))}")
    print("--------------------------------")
    print(f"Flash and tritt")
    print(f"max diff: {torch.max(tritt_out_flash - tritt_out)}")
    print(f"min diff: {torch.min(tritt_out_flash - tritt_out)}")
    print(f"mean abs diff: {torch.mean(torch.abs(tritt_out_flash - tritt_out))}")
