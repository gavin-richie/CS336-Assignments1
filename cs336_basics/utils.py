from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, BinaryIO, IO
from os import PathLike




def cross_entropy(inputs: Tensor, targets: Tensor)->Tensor:
    # assert (
    #         inputs.shape[-2] == targets.shape[-1]
    # ), f"inputs.shape[-2] {inputs.shape[-2]} != targets.shape[-1] {targets.shape[-1]}"
    # log_probs = inputs.float()
    # log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
    #
    # batch_idx = torch.arange(inputs.size(0), device=inputs.device)
    # loss = -log_probs[batch_idx, targets].mean()
    #
    # return loss.to(inputs.dtype)
    batch_size, vocab_size = inputs.size()
    """
        计算交叉熵损失，处理批量输入并确保数值稳定性。

        参数：
            inputs (Tensor): 未归一化的 logits，形状为 (batch_size, vocab_size)
            targets (Tensor): 目标类别索引，形状为 (batch_size,)

        返回：
            Tensor: 批量平均交叉熵损失，标量张量
        """
    # 减去最大值以提高数值稳定性
    max_logits, _ = torch.max(inputs, dim=-1, keepdim=True)
    shifted_logits = inputs - max_logits

    # 计算 log(sum(exp(shifted_logits)))，即 log-softmax 的分母
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # 收集目标位置的 logits
    target_logits = inputs[torch.arange(batch_size), targets]
    # 原公式-log{softmax(logit_i)[target_{i+1}]
    # 对softmax求log，对数指数消除，对数相除 对应除式相减 ：-logits[target_i] + log(sum(exp(logits_i)))
    # 计算交叉熵损失：-target_logit + log_sum_exp
    loss = -(target_logits - max_logits.squeeze(-1)) + log_sum_exp

    # 返回批量平均损失
    return loss.mean()

def get_batch(
    dataset: NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中采样语言模型的输入序列和对应的下一标记目标。

    参数：
        dataset (NDArray): 1D NumPy 数组（或 np.memmap），包含整数标记 ID
        batch_size (int): 批量大小(序列个数）
        context_length (int): 每个序列的上下文长度
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'）

    返回：
        Tuple[torch.Tensor, torch.Tensor]: 两个形状为 (batch_size, context_length) 的 LongTensor，
            第一个是输入序列，第二个是对应的下一标记目标
    """
    # 验证输入
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正值，得到 {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length 必须为正值，得到 {context_length}")
    if len(dataset) < context_length + 1:
        raise ValueError(f"数据集长度 {len(dataset)} 小于 context_length + 1 = {context_length + 1}")

    # 随机选择 batch_size 个起始索引
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # 构建输入序列和目标序列
    inputs = np.stack([dataset[i:i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])

    # 转换为 PyTorch 张量并移动到指定设备
    inputs_tensor = torch.LongTensor(inputs).to(device)
    targets_tensor = torch.LongTensor(targets).to(device)

    return inputs_tensor, targets_tensor


def save_checkpoint(model: nn.Module,optimizer: torch.optim.Optimizer,iteration:int,out:Union[str,PathLike,BinaryIO,IO[bytes]])->None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点加载模型权重、优化器状态，并返回迭代次数。

    参数：
        src (Union[str, PathLike, BinaryIO, IO[bytes]]): 检查点路径或文件对象
        model (nn.Module): 要恢复的模型
        optimizer (Optimizer): 要恢复的优化器

    返回：
        int: 保存的迭代次数
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
