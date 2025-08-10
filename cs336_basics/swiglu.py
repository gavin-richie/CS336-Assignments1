import torch
import torch.nn as nn
import torch.nn.init as init
from torch import FloatTensor

def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the SiLU (Swish) activation function: SiLU(x) = x * sigmoid(x).

    Args:
        x (torch.Tensor): Input tensor of any shape

    Returns:
        torch.Tensor: Output tensor of the same shape
    """
    return x * torch.sigmoid(x)

def glu(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """
    Apply the Gated Linear Unit (GLU): GLU(x, W1, W3) = SiLU(W1 x) * (W3 x).

    Args:
        x (torch.Tensor): Input tensor of shape (..., d_model)
        w1 (torch.Tensor): First linear weight, shape (d_ff, d_model)
        w3 (torch.Tensor): Third linear weight, shape (d_ff, d_model)

    Returns:
        torch.Tensor: Output tensor of shape (..., d_ff)
    """
    return SiLU(x @ w1.t()) * (x @ w3.t())

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU feed-forward network.

        Args:
            d_model (int): Hidden dimension of the model
            device (torch.device | None): Device to store the parameters on (default: None)
            dtype (torch.dtype | None): Data type of the parameters (default: None)
        """
        super().__init__()
        # Compute d_ff as 8/3 * d_model, rounded up to multiple of 64
        # d_ff = int((8/3) * d_model)
        # d_ff = ((d_ff + 63) // 64) * 64  # Round up to nearest multiple of 64

        # Initialize weight matrices
        self.w1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )

        # Initialize weights with truncated normal distribution
        for w in [self.w1, self.w2, self.w3]:
            in_dim, out_dim = w.shape
            std = (2.0 / (in_dim + out_dim)) ** 0.5
            init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor with SwiGLU: W2 (SiLU(W1 x) * W3 x).

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        # Apply GLU: SiLU(W1 x) * W3 x
        gate = glu(x, self.w1, self.w3)
        # Apply W2
        return gate @ self.w2.t()

class FFN(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        """
        Construct the Feed-Forward Network using SwiGLU.

        Args:
            d_model (int): Hidden dimension of the model
            device (torch.device | None): Device to store the parameters on (default: None)
            dtype (torch.dtype | None): Data type of the parameters (default: None)
        """
        super().__init__()
        self.swiglu = SwiGLU(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor with the SwiGLU-based FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        return self.swiglu(x)

def run_swiglu(
    d_model: int,
    w1: FloatTensor,
    w2: FloatTensor,
    w3: FloatTensor,
    in_features: FloatTensor,
) -> FloatTensor:
    """
    Given the weights of a SwiGLU layer, apply SwiGLU to a batch of input features.

    Args:
        d_model (int): The size of the input/output dimension
        w1 (FloatTensor): First linear weight, shape (d_ff, d_model)
        w2 (FloatTensor): Second linear weight, shape (d_model, d_ff)
        w3 (FloatTensor): Third linear weight, shape (d_ff, d_model)
        in_features (FloatTensor): Input features, shape (..., d_model)

    Returns:
        FloatTensor: Output features, shape (..., d_model)
    """
    d_ff = w1.shape[0]
    assert w1.shape == (d_ff, d_model), f"Expected w1 shape {(d_ff, d_model)}, got {w1.shape}"
    assert w2.shape == (d_model, d_ff), f"Expected w2 shape {(d_model, d_ff)}, got {w2.shape}"
    assert w3.shape == (d_ff, d_model), f"Expected w3 shape {(d_ff, d_model)}, got {w3.shape}"
    assert in_features.shape[-1] == d_model, f"Expected last dim {d_model}, got {in_features.shape[-1]}"

    swiglu = SwiGLU(d_model, device=w1.device, dtype=w1.dtype)
    swiglu.load_state_dict({"w1": w1, "w2": w2, "w3": w3})
    return swiglu(in_features)