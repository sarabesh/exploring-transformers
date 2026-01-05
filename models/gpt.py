
import torch.nn as nn

from layers.embedding import EmbeddingLayer
from blocks.transformer import TransformerBlock

class GPT(nn.Module): # Generative Pre-trained Transformer
    def __init__(self, vocab_size, embed_size, num_blocks=16,heads=8):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(embed_size, heads))
        self.out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x): # x shape: (batch_size, seq_length), input token IDs and output logits for each token in vocabulary
        x = self.embedding(x)  # Convert token IDs to embeddings
        for block in self.transformer_blocks:
            x = block(x)  # Pass through each transformer block
        x = self.out(x)  # Final linear layer to get logits for each token in vocabulary
        return x # shape: (batch_size, seq_length, vocab_size)