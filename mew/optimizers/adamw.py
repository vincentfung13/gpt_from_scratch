import math
from typing import Union, List, Dict, Optional
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Union[List, Dict],
        lr: float,
        weight_decay: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}!")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # Fetch group hyper params
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay, eps = group["weight_decay"], group["eps"]

            # Update params
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                # Fetch current step
                t = self.state[p].get("t", 1)

                # Fetch and update moment_1 (m)
                m = self.state[p].get(
                    "adamw.m", torch.zeros_like(p.data, device=p.data.device)
                )
                m = beta_1 * m + (1 - beta_1) * grad

                # Fetch and update moment_2 (v)
                v = self.state[p].get(
                    "adamw.v", torch.zeros_like(p.data, device=p.data.device)
                )
                v = beta_2 * v + (1 - beta_2) * grad.square()

                # Adjust learning rate and update params using adjusted learning rate and raw moments
                alpha_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))
                p.data -= alpha_t * m / (v.sqrt() + eps)

                # Apply weight decay
                p.data -= lr * weight_decay * p.data

                # Update state
                self.state[p].update({"t": t + 1, "adamw.m": m, "adamw.v": v})

        return loss
