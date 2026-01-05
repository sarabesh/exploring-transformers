#code for rope positional embeddings

import torch
import torch.nn as nn
from typing import Tuple


def apply_rotary_emb(xq: torch.Tensor,xk: torch.Tensor,freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # reshape last dim to dim/2, 2 i.e to (b,t,h, dim/2, 2) from (b,t,h, dim) and view as complex numbers(the last dim of size 2 is real and imag parts) (b,t,h, dim/2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # operate rotation in float32 for higher precision
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # reshape freqs_cis (from (t,dim/2) to (1,t,1,dim/2)) to match x(b,t,h,dim/2) for broadcasting
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2) #rotates in complex plane by multiplying with complex number of magnitude 1 and angle = position dependent freq theta_i, then view back as real and flatten last two dims to get back (b,t,h,dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2) #dim/2 in complex,dim/2,2 in real, flatten to dim 
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(theta=10000.0, dim=512, end=2048):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))   # base freq per pair, from 1 towards zero, 10000**(2i/dim) where i=0..dim/2 and reciprocal, giving 1000**(-2i/dim). Similar to sinusoidal pos emb
    t = torch.arange(end, dtype=torch.float32)                        # positions 0..end-1, 0 to seq_len-1
    freqs = torch.outer(t, freqs)                                     # theta= product t and freq, m.alpha_i where m=position, i=emb dim index/2 shape (end, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)             # complex numbers with magnitude 1 and angle freqs, e^(i*theta) = cos theta + i sin theta

    return freqs_cis  # (end, dim/2) returns table of complex numbers, rows = positions, cols = emb dim pair


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor): #reshape freqs_cis to match x for broadcasting, x is complex view of head tensor
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # x is complex view of head tensor, x.shape[1] seq len, x.shape[-1] dim/2
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # shape with 1s except seq len and dim/2
    return freqs_cis.view(*shape) # reshape freqs_cis to shape for broadcasting with x


if __name__=="__main__":
    dim = 512
    seq_len = 128
    batch_size = 2

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    freqs_cis = precompute_freqs_cis(theta=10000.0, dim=dim, end=seq_len)
    print(freqs_cis)