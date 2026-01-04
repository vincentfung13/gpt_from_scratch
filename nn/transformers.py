from einops import einsum, rearrange

import torch
import torch.nn as nn
import torch.nn.init as init

from gpt_from_scratch.nn.layers import RMSNorm, SwiGLU
from gpt_from_scratch.nn.functionals import scaled_dot_product
from gpt_from_scratch.nn.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        # RoPE related params
        theta: float,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = CausalMultiHeadSelfAttn(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch seq_len d_model)
        x += self.attn(self.ln1(x))
        x += self.ffn(self.ln2(x))
        return x


class CausalMultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        # Params for RoPE
        theta: float,
        max_seq_len: int,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        # d_q == d_k == d_v
        self.d_k = d_model // num_heads

        # Init weights as (num_heads d_model d_model // num_heads)
        # and d_q == d_k == d_v
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.output_proj = nn.Parameter(torch.empty(d_model, d_model))

        std = 1.0 / d_model
        init.trunc_normal_(self.q_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.k_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.v_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.output_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)

        # Init rope
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        # x: (... seq_len d_model)

        # Reshape weights for multi-head
        # Weights are (num_heads * d_k, d_model), reshape to (num_heads, d_k, d_model)
        q_proj = rearrange(
            self.q_proj,
            "(num_heads d_k) d_model -> num_heads d_k d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )
        k_proj = rearrange(
            self.k_proj,
            "(num_heads d_k) d_model -> num_heads d_k d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )
        v_proj = rearrange(
            self.v_proj,
            "(num_heads d_k) d_model -> num_heads d_k d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )

        # Compute Q, K, V
        q = einsum(
            q_proj,
            x,
            "num_heads d_k d_model, ... seq_len d_model -> ... num_heads seq_len d_k",
        )
        k = einsum(
            k_proj,
            x,
            "num_heads d_k d_model, ... seq_len d_model -> ... num_heads seq_len d_k",
        )
        v = einsum(
            v_proj,
            x,
            "num_heads d_k d_model, ... seq_len d_model -> ... num_heads seq_len d_k",
        )

        # Apply RoPE to q and k
        if token_positions is None:
            # Default to absolute positions (range(seq_len))
            seq_len = x.size(-2)
            token_positions = torch.arange(seq_len, device=x.device)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Create causal mask
        # Mask shape: (seq_len, seq_len)
        # True for allowed positions (i >= j), False elsewhere
        seq_len = x.size(-2)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # Compute scaled dot product
        # agg_v -> (... seq_len num_heads * d)
        agg_v = scaled_dot_product(q, k, v, mask)
        agg_v = rearrange(agg_v, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        # Cast back to d_model
        output = einsum(
            # agg_v, self.output_proj, "... seq_len d_in, d_out d_in -> ... seq_len d_out"
            self.output_proj,
            agg_v,
            "d_out d_in, ... seq_len d_in -> ... seq_len d_out",
        )

        return output
