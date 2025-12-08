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

import os
import torch
import torch.nn as nn
from einops import rearrange, repeat


class MultiHeadAttention(nn.Moduule):\
    #multi head attention module, input 
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.wq = nn.Linear(embed_size, embed_size) # weight matrices, shape: (embed_size, embed_size)
        self.wk = nn.Linear(embed_size, embed_size)
        self.wv = nn.Linear(embed_size, embed_size)


    def forward(self, x):
        #
        q = self.wq(x) # shape: (batch_size, seq_length, embed_size)
        k = self.wk(x)
        v = self.wv(x)


        q = rearrange(q, "b s (h d) -> (b h) s d", h=self.heads) # shape: (batch_size*heads, seq_length, head_dim)
        k = rearrange(k, "b s (h d) -> (b h) s d", h=self.heads)
        v = rearrange(v, "b s (h d) -> (b h) s d", h=self.heads)


        scores = attention(q,k,v) # shape: (batch_size*heads, seq_length, head_dim)
        scores = rearrange(scores, "(b h) s d -> b s (h d)", h=self.heads) # shape: (batch_size, seq_length, embed_size) concatenate all heads
        return scores




def attention(q, k, v): # shape: (batch_size*heads, seq_length, head_dim)
    d_k = k.size(-1)
    #q,k,v: (batch_size*heads, seq_length, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k) # shape: (batch_size*heads, seq_length, seq_length)
    weights = torch.nn.functional.softmax(scores, dim=-1) # shape: (batch_size*heads, seq_length, seq_length) Softmax along seq_length i.e each row
    output = torch.matmul(weights, v) # shape: (batch_size*heads, seq_length, head_dim)
    return output
    
 
# class embed(text):
#     #tokens = tokenize(text)
#     #embedding = embedding_matrix(tokens)
#     #return embedding

# class transformer_block(input_emb):
#     class multihead_attention(x):
#         #get Q,K,V for x
#         #split for each head
#         #do attention 
#         #concat all heads
#     #concat x with attention o/p
#     #layernorm
#     class feedforward(x):
#         #nn 2
#     # concat x with o/p
#     #return x

# class GPT(x):
#     #embed(x)
#     #position_enc(x)
#     #concat emb + pos
#     #x=transformer_block(x)*16
#     #linear_projection(x)