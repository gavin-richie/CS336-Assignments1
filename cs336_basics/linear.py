import torch
import math
import torch.nn as nn
import torch.nn.init as init
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module without bias.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None): Device to store the parameters on (default: None)
            dtype (torch.dtype | None): Data type of the parameters (default: None)
        """
        # Initialize weight matrix W with shape (out_features, in_features)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0/(in_features+out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x)->torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        # return torch.mm(x, self.weight)
        # return torch.matmul(x, self.weight.T)
        return x@self.weight.T

        # return torch.einsum("...d_in,d_out d_in->... d_out", x, self.weight)
        # return einsum(x,self.weight,"...d_in,d_out d_in->... d_out")
        # return x@self.weight.T
