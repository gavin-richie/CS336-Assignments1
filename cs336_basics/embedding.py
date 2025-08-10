import torch
import torch.nn as nn
from torch.nn import init

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0,b=-3.0)


    def forward(self, indices: torch.Tensor)->torch.Tensor:
        return self.embedding[indices]
