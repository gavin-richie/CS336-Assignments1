from collections.abc import Callable
import math
from typing import Optional, Iterable
import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float=1e-3) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params:Iterable, lr: float=1e-3,betas: tuple=(0.9,0.999),eps:float=1e8,weight_decay:float=0.0) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0<=betas[0]<1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0<=betas[1]<=1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if eps <=0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay<0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self,closure: Optional[Callable]=None)->None:
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state)==0:
                    # 初始化状态
                    state["t"] = 1  # 迭代次数
                    state["m"] = torch.zeros_like(p.data) # 第一阶矩向量
                    state["v"] = torch.zeros_like(p.data) # 第二阶矩向量

                t = state["t"]
                m = state["m"]
                v = state["v"]
                # 更新一阶动量：m = β1 * m + (1 - β1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶动量：v = β2 * v + (1 - β2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 使用学习率和动量第一次更新参数
                denom = v.sqrt().add_(eps) # 计算分母
                p.data.addcdiv_(m, denom,value = -lr_t) # m*-lr_t/denom
                # 使用权重衰减第二次更新参数
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr*weight_decay)
                # 更新迭代次数
                state['t'] += 1


def lr_cosine_schedule(t:int, max_lr:float, min_lr:float,warm_up: int,ct:int):
    """

    :param t: current step t / the current iteration
    :param max_lr: the maximum learning rate -> alpha_max
    :param min_lr: the minimum learning rate -> alpha_min
    :param warm_up: the number of warm-up iterations->T_w
    :param ct: the number of cosine annealing iterations->T_c
    :return:
    """
    if t < warm_up:
        alpha_t = t/warm_up * max_lr
        return alpha_t
    if warm_up<=t<=ct:
        theta = math.cos((t - warm_up)*math.pi/(ct-warm_up))
        alpha_t = min_lr+1/2*(1+theta)*(max_lr-min_lr)
        return alpha_t
    else:
        alpha_t = min_lr
        return alpha_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    实现梯度裁剪，将所有参数梯度的 L2 范数限制在 max_l2_norm 内。

    参数：
        parameters (Iterable[torch.nn.Parameter]): 可迭代的模型参数
        max_l2_norm (float): 最大 L2 范数，必须为正值

    返回：
        None: 直接原地修改 parameter.grad
    """
    if max_l2_norm <= 0:
        raise ValueError(f"max_l2_norm 必须为正值，得到 {max_l2_norm}")

    # 收集所有有效梯度
    valid_grads = [p.grad for p in parameters if p.grad is not None]
    if not valid_grads:
        return

    # 计算总 L2 范数：sqrt(sum(g_i^2))
    l2_norm = torch.sqrt(torch.sum(torch.stack([torch.sum(g ** 2) for g in valid_grads])))

    # 检查是否需要裁剪
    eps = 1e-6
    if l2_norm > max_l2_norm:
        # 计算缩放因子
        scale = max_l2_norm / (l2_norm + eps)
        # 原地缩放每个梯度
        for g in valid_grads:
            g.mul_(scale)

if __name__ == '__main__':
    weights = torch.nn.Parameter(5*torch.randn((10,10)))
    optimizer = SGD([weights],lr=1e1)
    for t in range(200):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        print(f"epoch:{t}, loss={loss.item()}")
        loss.backward()
        optimizer.step()
