Triton implementation of [n-way-attention](https://github.com/Gusanidas/n-way-attention), with the ideas from Flash Attention extrapolated to n-way-attention.

The forward pass achieves up to 100x better performance than the naive pytorch implementation, and can process much larger sequence lengths.

The backward pass is work in progress. Current implementation breaks down for a head_dim >= 64 in an A100.

References:
[Flash Attention paper](https://arxiv.org/pdf/2205.14135)
[Umar Jamil Flash Attention](https://github.com/hkproj/triton-flash-attention)
[Triton's Tutorial Flash Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)
