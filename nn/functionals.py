import math
from einops import einsum

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


def scaled_dot_product(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    # q -> (batch ... seq_len_q d_k)
    # k -> (batch ... seq_len_k d_k)
    # v -> (batch ... seq_len_k d_v)
    # mask -> (seq_len seq_len)
    # returns output -> (batch ... seq_len_q d_v)
    qk_dot = einsum(
        q,
        k,
        "batch ... seq_len_q d_k, batch ... seq_len_k d_k -> batch ... seq_len_q seq_len_k",
    )
    qk_dot /= math.sqrt(q.size()[-1])

    # Before softmax, apply mask if specified
    if mask is not None:
        # Set very negative value (e.g., -1e9) to masked positions
        qk_dot = qk_dot.masked_fill(~mask, float("-inf"))

    # Reduce with V
    attn_weights = softmax(qk_dot, dim=-1)  # (batch ... seq_len_q seq_len_k)
    aggregated_v = einsum(
        attn_weights,
        v,
        "batch ... seq_len_q seq_len_k, batch ... seq_len_k d_v -> batch ... seq_len_q d_v",
    )

    return aggregated_v
