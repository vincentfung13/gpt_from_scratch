from einops import einsum, reduce

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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-05,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model))

        # Move to specified device
        if device is not None:
            self.g = self.g.to(device)

        # Cast to specified type
        if dtype is not None:
            self.g = self.g.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (b seq d)
        in_dtype = x.dtype

        # Upcast and compute RMS
        x_square = x.to(torch.float32).square()
        rms = (
            self.eps + reduce(x_square, "b seq d -> b seq ()", "sum") / self.d_model
        ).sqrt()

        # Divide by RMS and multiply by gain
        x = einsum(self.g, x / rms, "d, b seq d -> b seq d")

        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: str = None, dtype: torch.dtype = None
    ):
        super().__init__()
        # Create weight Tensor
        # w1 and w3 cast input to d_ff
        # w2 casts d_ff to d_model and produce output
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

        # Init parameters
        std = 2.0 / (d_model + d_ff)
        init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3 * std, b=3 * std)

        # Move to specified device
        if device is not None:
            self.w1 = self.w1.to(device)
            self.w2 = self.w2.to(device)
            self.w3 = self.w3.to(device)

        # Cast to specific types
        if dtype is not None:
            self.w1 = self.w1.to(dtype)
            self.w2 = self.w2.to(dtype)
            self.w3 = self.w3.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> ... d_model
        w1_x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_w1_x = w1_x / (1 + torch.exp(-w1_x))
        mul = einsum(silu_w1_x, w3_x, "... d_ff, ... d_ff -> ... d_ff")
        output = einsum(self.w2, mul, "d_model d_ff, ... d_ff -> ... d_model")
        return output
