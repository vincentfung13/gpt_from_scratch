import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.current_step = 0
        super().__init__(optimizer)

    def step(self):
        lr = get_cosine_annealing_lr(
            self.current_step,
            self.max_learning_rate,
            self.min_learning_rate,
            self.warmup_iters,
            self.cosine_cycle_iters,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_step += 1


def get_cosine_annealing_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # annealing_steps represents the number of cosine annealing iterations
    # Total steps = warmup_iters + annealing_steps
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        cos_rad = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(cos_rad)) * (
            max_learning_rate - min_learning_rate
        )
    else:
        # t > total_steps
        return min_learning_rate
