import torch
import torch.nn as nn
# from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement RMSNorm
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Hint: You can use torch.rsqrt to compute the reciprocal square root.
        rms =  torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * rms * self.weight
        return result.to(in_dtype)
