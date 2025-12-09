import os
import torch
from torch import nn

class EmbeddingLayer(nn.Module):
    # Initializes embedding layer with matrix of size (vocab_size, embedding_dim), to map token ids to dense vectors. This is different from linear layer, that it maps token id instead of doing matrix multiplication.
    # you can either do this, or use nn.Embedding from pytorch directly or use one hot encoding followed by linear layer(to simulate lookup behavior).
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim)) # Parameter are matrixes that can be optimized by optimzer in torch, i.e it tracks gradients

    def forward(self, input_ids):
        return self.weight[input_ids]

if __name__ == "__main__":
    embedding = EmbeddingLayer(vocab_size=256, embedding_dim=64)
    print (embedding(torch.tensor([1,2,3,4,5]))) # should print a matrix of size (5, 64) of randomly initialized values