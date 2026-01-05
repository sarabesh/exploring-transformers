import torch.nn as nn
from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm(x + attn_output)  # Residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm(x + ff_output)    # Residual connection and layer normalization
        return x