from einops import einsum

import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        # Create weight Tensor
        self.weights = nn.Parameter(torch.empty(out_features, in_features))

        # Init weights
        std = 2.0 / (in_features + out_features)
        init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

        # Move to specified device
        if device is not None:
            self.weights = self.weights.to(device)

        # Cast to specified type
        if dtype is not None:
            self.weights = self.weights.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            self.weights, x, "out_feats in_feats, ... in_feats -> ... out_feats"
        )


class Embedding(nn.Module):
    def __init__(
        self,
        num_weights: int,
        embedding_dim: int,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        # Create embedding lookup table and initialization
        self.weights = nn.Parameter(torch.empty(num_weights, embedding_dim))
        init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

        # Move to specified device
        if device is not None:
            self.weights = self.weights.to(device)

        # Cast to specified type
        if dtype is not None:
            self.weights = self.weights.to(dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
