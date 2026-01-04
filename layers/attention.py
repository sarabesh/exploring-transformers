#let's try to learn attention, might revert to a notebook if required.

#autoregressive

#i/p

#*16
#transformer block
#mha
#attention block
#embedding -> Q,K,V -> softmax(Q.K).V
#concat
#mcp

#o/p

import torch
import torch.nn as nn
from einops import rearrange, repeat
from rope import apply_rotary_emb, precompute_freqs_cis
from embedding import EmbeddingLayer

class MultiHeadAttention(nn.Moduule):
    #multi head attention module, input 
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.wq = nn.Linear(embed_size, embed_size) # weight matrices, shape: (embed_size, embed_size)
        self.wk = nn.Linear(embed_size, embed_size) # linear layers, number of inputs to each neuron, number of neurons/output size, creating weights of size (in_features, out_features)
        self.wv = nn.Linear(embed_size, embed_size)


    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        q = self.wq(x) # shape: (batch_size, seq_length, embed_size)
        k = self.wk(x)
        v = self.wv(x)

        freqs_cis = precompute_freqs_cis(theta=10000.0, dim=self.head_dim, end=x.shape[1])
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = rearrange(q, "b s (h d) -> (b h) s d", h=self.heads) # shape: (batch_size*heads, seq_length, head_dim)
        k = rearrange(k, "b s (h d) -> (b h) s d", h=self.heads)
        v = rearrange(v, "b s (h d) -> (b h) s d", h=self.heads)


        scores = attention(q,k,v) # shape: (batch_size*heads, seq_length, head_dim)
        scores = rearrange(scores, "(b h) s d -> b s (h d)", h=self.heads) # shape: (batch_size, seq_length, embed_size) concatenate all heads
        return scores


def attention(q, k, v, causal=False): # shape: (batch_size*heads, seq_length, head_dim)
    d_k = k.size(-1)
    #q,k,v: (batch_size*heads, seq_length, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k) # shape: (batch_size*heads, seq_length, seq_length)
    if causal: #for llms
        causal_mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool().to(scores.device) # upper triangular matrix with 1s above diagonal
        scores = scores.masked_fill(causal_mask, float('-inf')) # mask out future positions
    weights = torch.nn.functional.softmax(scores, dim=-1) # shape: (batch_size*heads, seq_length, seq_length) Softmax along seq_length i.e each row
    output = torch.matmul(weights, v) # shape: (batch_size*heads, seq_length, head_dim)
    return output
    
