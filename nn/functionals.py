import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # x -> (... dim)

    # Compute exp(x) - subtract max first for stability
    dim_max = x.max(dim=dim, keepdim=True).values
    x -= dim_max
    exp_x = x.exp()

    # Compute exp_sum
    exp_sum = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / exp_sum
