Triton implementation of [n-way-attention](https://github.com/Gusanidas/n-way-attention), with some modifications and using the main ideas of tiling and online softmax from Flash Attention.

### Standard Attention Equations

$$S_{ij} = \sum_{d} Q_{id} \cdot K_{jd}$$

$$A_{ij} = \frac{\exp(S_{ij})}{\sum_{j'} \exp(S_{ij'})}$$

$$Z_{id} = \sum_{j} A_{ij} \cdot V_{jd}$$

The final output $Z \in \mathbb{R}^{B \times N \times T \times D}$ has the same shape as the input.

### Triple Dot Product

$$S_{ijq} = \sum_{h} K1_{ijh} \cdot K2_{ikh} \cdot Q_{iqh}$$


$$A_{iq,jk} = \frac{\exp(\alpha \cdot S_{iq,jk})}{\sum_{j'k'} \exp(\alpha \cdot S_{iq,j'k'})}$$

$$V_{ijkh} = \text{SiLU}(V1_{ijh}) \cdot V2_{ikh}$$

$$Z_{iqh} = \sum_{jk} A_{iq,jk} \cdot V'_{ih,jk}$$

The final output $Z \in \mathbb{R}^{B \times N \times T \times D}$ has the same shape as the input queries.

### Dot product sum


$$S_{istq} = \sum_{h} K2_{ith} \cdot Q_{iqh} + \sum_{h} K1_{ish} \cdot K2_{ith}$$

$$A_{iq,st} = \frac{\exp(\alpha \cdot S_{istq})}{\sum_{s't'} \exp(\alpha \cdot S_{is't'q})}$$

$$V_{isth} = \text{SiLU}(V1_{ish}) \cdot V2_{ith}$$

$$Z_{iqh} = \sum_{st} A_{iq,(st)} \cdot V_{isth}$$

The final output $Z \in \mathbb{R}^{B \times N \times T \times D}$ has the same shape as the input queries.



References:
[Flash Attention paper](https://arxiv.org/pdf/2205.14135)
[Umar Jamil Flash Attention](https://github.com/hkproj/triton-flash-attention)
[Triton's Tutorial Flash Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)
