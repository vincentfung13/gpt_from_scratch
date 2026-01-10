from collections.abc import Iterable
from einops import reduce

import torch


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-08
):
    grads = [p.grad.data for p in parameters if p.grad is not None]
    l2_norm = sum(
        [reduce(g.square(), "... -> ()", reduction="sum") for g in grads]
    ).sqrt()

    if l2_norm > max_l2_norm:
        scale_factor = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale_factor
