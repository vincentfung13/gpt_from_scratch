import math


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
