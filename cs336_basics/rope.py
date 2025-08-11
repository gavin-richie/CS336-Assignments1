import torch.nn as nn
import torch
from einops import rearrange, reduce, einsum
import math
from torch import Tensor
from jaxtyping import Float, Int


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None,dtype=None):
        """
        Initialize the Rotary Positional Embedding (RoPE) module.

        Args:
            theta (float): Scaling factor Θ for RoPE.
            d_k (int): Dimension of query/key vectors (must be even).
            max_seq_len (int): Maximum sequence length for pre-caching sin/cos.
            device (torch.device | None): Device to store buffers (default: None).
        """
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        # Generate position indices: [0, 1, ..., max_seq_len-1]
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # Generate frequency indices: [0, 0, 1, 1, ..., d_k/2-1, d_k/2-1]
        freq_idx = torch.arange(d_k // 2, dtype=torch.float32, device=device)
        freq_idx = freq_idx.repeat_interleave(2)  # Shape: (d_k,)

        # Compute angles: θ_{i,k} = i / Θ^((2k-1)/d)
        inv_freq = theta ** (-2 * freq_idx / d_k)  # Shape: (d_k,)
        angles = pos[:, None] * inv_freq[None, :]  # Shape: (max_seq_len, d_k)

        # Precompute cos and sin
        cos = torch.cos(angles)  # Shape: (max_seq_len, d_k)
        sin = torch.sin(angles)  # Shape: (max_seq_len, d_k)

        # Register as buffers (non-learnable)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Token positions of shape (..., seq_len).

        Returns:
            torch.Tensor: Rotated tensor of shape (..., seq_len, d_k).
        """
        seq_len = x.size(-2)
        d_k = x.size(-1)

        # Ensure token_positions is long type for indexing
        pos = token_positions.long()  # Shape: (..., seq_len)

        # Slice cos and sin using token_positions
        # Expand pos to match dimensions for gather
        pos = pos.unsqueeze(-1).expand(*pos.shape, d_k)  # Shape: (..., seq_len, d_k)
        cos = torch.gather(self.cos[:seq_len], 0, pos)  # Shape: (..., seq_len, d_k)
        sin = torch.gather(self.sin[:seq_len], 0, pos)  # Shape: (..., seq_len, d_k)

        # Split x into pairs: x_2k-1, x_2k
        x_even = x[..., 0::2]  # Shape: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # Shape: (..., seq_len, d_k/2)

        # Apply rotation: [x_2k-1, x_2k] -> [x_2k-1 * cos - x_2k * sin, x_2k * cos + x_2k-1 * sin]
        x_rotated_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        x_rotated_odd = x_even * sin[..., 0::2] + x_odd * cos[..., 0::2]

        # Interleave results to restore original shape
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).view(*x.shape)

        return x_rotated

    # def __init__(self, theta:float, d_k: int, max_seq_len: int, device=None, dtype=None):
    #     super().__init__()
    #     self.theta = theta
    #     self.d_k = d_k
    #     self.max_seq_len = max_seq_len
    #
    #     self.half_dim = d_k // 2
    #     freq_seq=torch.arange(self.half_dim,  device=device, dtype=dtype)
    #     inv_freq=1.0 / (self.theta ** (freq_seq / self.half_dim))
    #
    #     t=torch.arange(max_seq_len,dtype=torch.float32, device=device)
    #
    #     freqs=einsum(t,inv_freq,"i,j->i j")
    #     cos=torch.cos(freqs)
    #     sin=torch.sin(freqs)
    #     self.register_buffer("sin_cached", sin, persistent=False)
    #     self.register_buffer("cos_cached", cos, persistent=False)
    #
    # def forward(
    #         self,
    #         x: Float[Tensor, "... seq_len d_k"],
    #         token_positions: Int[Tensor, "... seq_len"],
    # ) -> Float[Tensor, "...  seq_len d_k"]:
    #     assert x.shape[-1] == self.d_k, f"x's last dim {x.shape[-1]} != d_k {self.d_k}"
    #     assert self.d_k % 2 == 0, "d_k must be even for RoPE"
    #
    #     in_type = x.dtype
    #     x = x.to(torch.float32)
    #
    #     # (... seq_len d_k) ->  (... seq_len d_pair 2) 2D-Tensor
    #     x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two=2)
    #
    #     # cos/sin tensor build
    #     cos = self.cos_cached[token_positions]
    #     sin = self.sin_cached[token_positions]
    #
    #     cos = rearrange(cos, "... s d -> ... 1 s d")
    #     sin = rearrange(sin, "... s d -> ... 1 s d")
    #
    #     x1, x2 = x_pair.unbind(dim=-1)
    #     rot1 = x1 * cos - x2 * sin
    #     rot2 = x1 * sin + x2 * cos
    #     x_rot = torch.stack((rot1, rot2), dim=-1)
    #
    #     out = rearrange(x_rot, "... s d two -> ... s (d two)", two=2)
    #
    #     return out.to(in_type)
