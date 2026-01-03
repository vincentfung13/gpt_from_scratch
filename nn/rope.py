import math
from einops import rearrange, einsum

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        assert self.d_k % 2 == 0
        self.max_seq_len = max_seq_len

        # Init rotary matrices
        # R -> (max_seq_len k 2 2)
        self.R = []
        for i in range(max_seq_len):
            R_i = []
            for k in range(1, self.d_k // 2 + 1):
                theta_i_k = i / (self.theta ** (2 * k - 2) / self.d_k)
                R_i_k = [
                    [math.cos(theta_i_k), -math.sin(theta_i_k)],
                    [math.sin(theta_i_k), math.cos(theta_i_k)],
                ]
                R_i.append(R_i_k)
            self.R.append(R_i)

        # Convert to tensor
        self.R = torch.tensor(self.R, dtype=torch.float32)

        if device is not None:
            self.R = self.R.to(device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x -> (... seq_len d_k)
        # token_positions -> (... seq_len)

        # 1. Grab R -> (... seq_len k 2 2)
        R = self.R[token_positions]

        # 2. Rearrange x (... seq_len d_k -> ... seq_len k 2)
        x = rearrange(
            x, "... seq_len (k d2) -> ... seq_len k d2", k=self.d_k // 2, d2=2
        )

        # 3. Mul embedding
        x = einsum(R, x, "... seq_len k d2 d2, ... seq_len k d2 -> ... seq_len k d2")

        # 4. Restore x (... seq_len k 2 -> ... seq_len d_k)
        x = rearrange(x, "... seq_len k d2 -> ... seq_len (k d2)")

        return x
