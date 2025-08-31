import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Dict
from cs336_basics.attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import FFN, glu
from tests.conftest import vocab_size
# from torch.nn.functional import softmax
from cs336_basics.softmax import softmax
from jaxtyping import Float, Int

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int,theta:float,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)

    def forward(self,x:Tensor, token_positions:Optional[Tensor]=None) -> Tensor:
        x_norm = self.ln1(x)
        # if 'ln1.weight' in weights.keys():
        #     x_norm = x_norm * weights['ln1.weight']

        attn_out = self.attn(x_norm,
                             # q_proj_weight=weights.get('attn.q_proj.weight'),
                             # k_proj_weight=weights.get('attn.k_proj.weight'),
                             # v_proj_weight=weights.get('attn.v_proj.weight'),
                             # o_proj_weight=weights.get('attn.output_proj.weight'),
                             token_positions=token_positions)
        x = x + attn_out
        x_norm = self.ln2(x)
        # if 'ln2.weight' in weights.keys():
            # x_norm = x_norm * weights['ln2.weight']

        # if 'ffn.w1.weight' in weights.keys():
        #     gate = glu(x_norm, weights.get('ffn.w1.weight'), weights.get('ffn.w3.weight'))
        #     ffn_out = gate @ weights.get('ffn.w2.weight').t()
        # else:
        #     ffn_out = self.ffn(x_norm)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int, num_heads:int, d_ff:int, theta:float, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype)
            for _ in range(num_layers)])

        self.ln_final = RMSNorm(d_model,device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size,bias=False, device=device,dtype=dtype)

    def forward(self, in_indices:Tensor) -> Tensor:
        x = self.token_embedding(in_indices)
        batch_size, seq_len = in_indices.shape

        # for layer in self.layers:
        #     x = layer(x, token_positions=token_positions)

        # 通过 Transformer 层
        # for i in range(self.num_layers):
        #     layer_prefix = f'layers.{i}.'
            # layer_weights = {k[len(layer_prefix):]: weights[k] for k in weights if k.startswith(layer_prefix)}
            # x = self.layers[i](x, layer_weights)
        for block in self.layers:
            x = block(x)

        # 最终 RMSNorm
        x_final = self.ln_final(x)
        # x_final = x * weights['ln_final.weight']

        # LM Head
        logits = self.lm_head(x_final)
        # logits = x_final @ weights['lm_head.weight'].t()

        return logits

    def print_trainable_parameters(self, verbose=False):
        """
        Prints the number of trainable parameters in the model.
        If verbose=True, also prints the name and shape of each trainable parameter.
        """
        total_params = 0
        trainable_params = 0

        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()  # Number of elements in the parameter tensor
                total_params += param_count
                trainable_params += param_count
                if verbose:
                    print(f"  {name}: shape={param.shape}, params={param_count}")
            else:
                total_params += param.numel()
                if verbose:
                    print(f"  {name}: shape={param.shape}, params={param.numel()} (non-trainable)")

        print(f"\nSummary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        return trainable_params

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        从语言模型生成文本序列，支持温度缩放、top-k 采样和 EOS 标记停止。

        参数：
            prompt (Tensor): 输入提示序列，形状为 (1, seq_len) 或 (seq_len,)，包含整数标记 ID
            max_new_tokens (int): 最大生成标记数
            temperature (float): 温度参数，控制 softmax 分布平滑度，默认 1.0
            top_k (Optional[int]): 若提供，仅从概率最高的 top_k 个词汇采样
            eos_token_id (Optional[int]): 若提供，生成此 ID 时停止

        返回：
            Tensor: 生成的标记序列，形状为 (1, new_tokens)，包含新生成的标记

        抛出：
            ValueError: 如果输入参数无效（例如 temperature <= 0 或 max_new_tokens < 0）
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token 表示所有样本中、最后一个 token 位置，模型对所有 32000 个可能 token 的 logit 分数
            next_token_logits = logits[:, -1] #【batch_size,1,32000】
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            # get max probability token vocab-index from multinomial distribution
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids
        # 输入验证
        # if temperature <= 0:
        #     raise ValueError(f"temperature 必须为正值，得到 {temperature}")
        # if max_new_tokens < 0:
        #     raise ValueError(f"max_new_tokens 必须非负，得到 {max_new_tokens}")
        # if top_k is not None and (top_k <= 0 or top_k > self.vocab_size):
        #     raise ValueError(f"top_k 必须在 (0, {self.vocab_size}]，得到 {top_k}")
        # if eos_token_id is not None and (eos_token_id < 0 or eos_token_id >= self.vocab_size):
        #     raise ValueError(f"eos_token_id 必须在 [0, {self.vocab_size})，得到 {eos_token_id}")
        #
        # # 规范化提示序列为 (1, seq_len)
        # if prompt.dim() == 1:
        #     prompt = prompt.unsqueeze(0)
        # if prompt.dim() != 2 or prompt.size(0) != 1:
        #     raise ValueError(f"prompt 形状必须为 (1, seq_len) 或 (seq_len,)，得到 {prompt.shape}")
        #
        # # 初始化生成序列
        # generated = prompt.clone().to(prompt.device)
        # original_seq_len = generated.size(-1)
        #
        # # 逐个生成标记
        # for _ in range(max_new_tokens):
        #     # 截取最后 context_length 个标记，确保不超过模型上下文长度
        #     input_seq = generated[:, -self.context_length:] if generated.size(-1) > self.context_length else generated
        #
        #     # 获取模型输出的 logits
        #     logits = self.forward(input_seq)  # 形状: (1, seq_len, vocab_size)
        #     next_token_logits = logits[:, -1, :]  # 形状: (1, vocab_size)
        #
        #     # 应用温度缩放
        #     scaled_logits = next_token_logits / temperature
        #
        #     # 应用 top-k 采样
        #     if top_k is not None:
        #         # 获取 top-k 值和索引
        #         topk_values, topk_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
        #         # 创建掩码，将非 top-k 的 logits 置为 -inf
        #         mask = torch.ones_like(scaled_logits, dtype=torch.bool)
        #         mask.scatter_(dim=-1, index=topk_indices, value=False)
        #         scaled_logits = scaled_logits.masked_fill(mask, float("-inf"))
        #
        #     # 计算概率分布
        #     probabilities = softmax(scaled_logits, dim=-1)
        #
        #     # 采样下一个标记
        #     next_token = torch.multinomial(probabilities, num_samples=1)  # 形状: (1, 1)
        #
        #     # 检查是否生成 EOS 标记
        #     if eos_token_id is not None and next_token.item() == eos_token_id:
        #         break
        #
        #     # 追加新标记到序列
        #     generated = torch.cat([generated, next_token], dim=-1)
        #
        # # 返回新生成的标记
        # return generated[:, original_seq_len:]