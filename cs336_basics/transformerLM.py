from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from cs336_basics.attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import FFN, glu


class TransformerBlock(nn.Module):
    def __init__(self, d_model:int,num_heads:int,d_ff:int,max_seq_len: int,theta:float=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,theta=theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_ff, d_model)

    def forward(self, x: Tensor, weights: Dict[str, Tensor], token_positions: Optional[Tensor] = None) -> Tensor:
        x_norm = self.ln1(x)
        if 'ln1.weight' in weights.keys():
            x_norm = x_norm * weights['ln1.weight']

        attn_out = self.attn(x_norm,
                             q_proj_weight=weights.get('attn.q_proj.weight'),
                             k_proj_weight=weights.get('attn.k_proj.weight'),
                             v_proj_weight=weights.get('attn.v_proj.weight'),
                             o_proj_weight=weights.get('attn.output_proj.weight'),
                             token_positions=token_positions)
        x = x + attn_out
        x_norm = self.ln2(x)
        if 'ln2.weight' in weights.keys():
            x_norm = x_norm * weights['ln2.weight']

        if 'ffn.w1.weight' in weights.keys():
            gate = glu(x_norm, weights.get('ffn.w1.weight'), weights.get('ffn.w3.weight'))
            ffn_out = gate @ weights.get('ffn.w2.weight').t()
        else:
            ffn_out = self.ffn(x_norm)
        # ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: Tensor, weights: Dict[str, Tensor]) -> Tensor:
        # x = self.token_embedding(in_indices)
        # if 'token_embedding_weight' in weights.keys():
        #     x = x + weights['token_embedding_weight']
        x = F.embedding(in_indices, weights['token_embeddings.weight'])
        batch_size, seq_len = in_indices.shape

        # for layer in self.layers:
        #     x = layer(x, token_positions=token_positions)

        # 通过 Transformer 层
        for i in range(self.num_layers):
            layer_prefix = f'layers.{i}.'
            layer_weights = {k[len(layer_prefix):]: weights[k] for k in weights if k.startswith(layer_prefix)}
            x = self.layers[i](x, layer_weights)
        # for block in self.layers:
        #     x = block(x)

        # 最终 RMSNorm
        x = self.ln_final(x)
        x_final = x * weights['ln_final.weight']

        # LM Head
        # logits = self.lm_head(x_final) # 直接使用linear层lm_head.weight的权重需要提前加载
        logits = x_final @ weights['lm_head.weight'].t()
        # logits = torch.matmul(logits, weights['lm_head.weight'].transpose(-2, -1))

        return logits

