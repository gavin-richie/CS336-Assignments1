import torch
from torch import Tensor
from jaxtyping import Float
# from einops import reduce

def softmax(in_features: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """
    Apply softmax to the specified dimension of the input tensor with numerical stability.

    Args:
        in_features (FloatTensor): Input tensor of arbitrary shape.
        dim (int): Dimension to apply softmax to.

    Returns:
        FloatTensor: Tensor with the same shape as in_features, with softmax applied to the specified dim.
    """
    # Subtract the maximum value along the specified dimension for numerical stability
    max_values = torch.max(in_features, dim=1,keepdim=True)[0]
    exp_inputs = torch.exp(in_features - max_values)

    # Compute the sum of exponentials along the specified dimension
    sum_exp = torch.sum(exp_inputs, dim=1, keepdim=True)

    # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    result = exp_inputs / sum_exp
    return result

    # if dim < 0:
    #     dim += in_features.ndim
    #
    # in_type = in_features.dtype
    # in_features = in_features.to(torch.float64)
    #
    # perm = list(range(in_features.ndim))
    # perm[dim], perm[-1] = perm[-1], perm[dim]
    # x_moved = in_features.permute(*perm)
    #
    # x_max = reduce(x_moved, "... n -> ... 1", "max")
    # x_exp = (x_moved - x_max).exp()
    #
    # x_enom = reduce(x_exp, "... n -> ... 1", "sum")
    #
    # out_moved = x_exp / x_enom
    #
    # out = out_moved.permute(*perm)
    #
    # return out.to(in_type)
