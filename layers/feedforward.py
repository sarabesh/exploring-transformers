import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, embed_size*4)
        self.fc2 = nn.Linear(embed_size*4, embed_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x