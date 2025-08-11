import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                Q:Tensor,
                K:Tensor,
                V:Tensor,
                mask:Optional[Tensor]=None)->Tensor:
        d_k=Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2))/torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) # 点积注意力和随机多头注意力的mask填充是相反的
            # 点积注意力的mask如果是Ture表示权重不变，随机多头注意力的mask如果是True表示负无穷
            # mask = mask.float()
            # mask=(1-mask)*-1e9
            # scores = scores+mask
        attn_weights = scores.softmax(dim=-1)
        # attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multihead Self-Attention module without RoPE.

        Args:
            d_model (int): Model dimension (input and output dimension)
            num_heads (int): Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = self.d_k  # Following Vaswani et al., d_v = d_k

        self.attention = ScaledDotProductAttention()

    def forward(
            self,
            x: Tensor,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            o_proj_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for multi-head self-attention with causal masking.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            q_proj_weight (Optional[Tensor]): Query projection weights of shape (d_model, d_model)
            k_proj_weight (Optional[Tensor]): Key projection weights of shape (d_model, d_model)
            v_proj_weight (Optional[Tensor]): Value projection weights of shape (d_model, d_model)
            o_proj_weight (Optional[Tensor]): Output projection weights of shape (d_model, d_model)
            mask (Optional[Tensor]): Optional additional mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        test_flag = True
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = self.d_model // self.num_heads
        batch_size, seq_len, d_in = x.size()[-3:]

        # Linear projections
        Q = torch.matmul(x, q_proj_weight.transpose(-2, -1))  # (..., seq_len, d_model)
        K = torch.matmul(x, k_proj_weight.transpose(-2, -1))  # (..., seq_len, d_model)
        V = torch.matmul(x, v_proj_weight.transpose(-2, -1))  # (..., seq_len, d_model)

        # Reshape for multi-head: (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(-3, -2)
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(-3, -2)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(-3, -2)

        # Compute attention logits
        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (..., num_heads, seq_len, seq_len)

        # Create and apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))

        # Compute attention weights and apply to values
        attn_weights = attn_logits.softmax(dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (..., num_heads, seq_len, head_dim)


        # Reshape back: (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        attn_out = attn_out.transpose(-3, -2).contiguous().view(batch_size, seq_len, self.d_model)


        # Final output projection
        out = torch.matmul(attn_out, o_proj_weight.transpose(-2, -1))  # (..., seq_len, d_model)

        return out


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize Multihead Self-Attention with Rotary Position Embeddings (RoPE).

        Args:
            d_model (int): Model dimension (input and output dimension)
            num_heads (int): Number of heads to use in multi-headed attention
            theta (float): RoPE base frequency parameter
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = self.d_k  # Following Vaswani et al., d_v = d_k

        self.attention = ScaledDotProductAttention()
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))

    def _apply_rope(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply Rotary Position Embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, d_k)
            positions (Optional[Tensor]): Token positions of shape (..., seq_len)

        Returns:
            Tensor: RoPE-applied tensor of same shape as input
        """
        batch_size, _, seq_len, _ = x.size()

        # Use provided positions or default to 0, 1, ..., seq_len-1
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)  # (seq_len,)

        # Ensure positions have correct shape for broadcasting
        positions = positions.view(-1, seq_len, 1)  # (..., seq_len, 1)
        angles = positions * self.inv_freq.to(x.device)  # (..., seq_len, d_k/2)

        # Compute sin and cos for rotations
        sin_angles = torch.sin(angles)  # (..., seq_len, d_k/2)
        cos_angles = torch.cos(angles)  # (..., seq_len, d_k/2)

        # Split x into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split along d_k dimension

        # Apply rotation: x1' = x1 * cos - x2 * sin, x2' = x1 * sin + x2 * cos
        x1_rot = x1 * cos_angles - x2 * sin_angles
        x2_rot = x1 * sin_angles + x2 * cos_angles

        # Interleave rotated pairs
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).view_as(x)

        return x_rot

    def forward(
            self,
            x: Tensor,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            o_proj_weight: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            token_positions: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for multi-head self-attention with RoPE and causal masking.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            q_proj_weight (Optional[Tensor]): Query projection weights of shape (d_model, d_model)
            k_proj_weight (Optional[Tensor]): Key projection weights of shape (d_model, d_model)
            v_proj_weight (Optional[Tensor]): Value projection weights of shape (d_model, d_model)
            o_proj_weight (Optional[Tensor]): Output projection weights of shape (d_model, d_model)
            mask (Optional[Tensor]): Optional additional mask of shape (batch_size, seq_len, seq_len)
            token_positions (Optional[Tensor]): Token positions of shape (..., seq_len)

        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()

        # Use provided weights or identity matrices
        W_q = q_proj_weight if q_proj_weight is not None else torch.eye(d_model, device=x.device)
        W_k = k_proj_weight if k_proj_weight is not None else torch.eye(d_model, device=x.device)
        W_v = v_proj_weight if v_proj_weight is not None else torch.eye(d_model, device=x.device)
        W_o = o_proj_weight if o_proj_weight is not None else torch.eye(d_model, device=x.device)

        # Linear projections in a single matrix multiply
        Q = torch.matmul(x, W_q.t())  # (batch_size, seq_len, d_model)
        K = torch.matmul(x, W_k.t())  # (batch_size, seq_len, d_model)
        V = torch.matmul(x, W_v.t())  # (batch_size, seq_len, d_model)

        # Reshape for multi-head: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # Apply RoPE to Q and K (but not V)
        Q = self._apply_rope(Q, token_positions)
        K = self._apply_rope(K, token_positions)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

        # Combine causal mask with optional input mask
        if mask is not None:
            causal_mask = causal_mask & mask

        # Apply scaled dot-product attention
        output = self.attention(Q, K, V, ~causal_mask)  # (batch_size, num_heads, seq_len, d_v)

        # Reshape back: (batch_size, num_heads, seq_len, d_v) -> (batch_size, seq_len, num_heads * d_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = torch.matmul(output, W_o.t())  # (batch_size, seq_len, d_model)

        return output