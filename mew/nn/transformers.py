from einops import einsum, rearrange
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from mew.nn.layers import RMSNorm, SwiGLU
from mew.nn.functionals import scaled_dot_product
from mew.nn.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        # RoPE related params
        theta: float,
        max_seq_len: int = 4096,
        # GQA configuration
        num_groups: int = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = CausalMultiHeadSelfAttn(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            num_groups=num_groups,
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch seq_len d_model)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
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
        # GQA configuration
        num_groups: int = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_model = d_model
        self.d_head = d_model // num_heads

        if num_groups is None:
            # Assume full MHA
            self.d_head = d_model // num_heads

            # Init weights as (num_heads d_model d_model // num_heads)
            self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
            self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
            self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
            self.output_proj = nn.Parameter(torch.empty(d_model, d_model))
        else:
            # Use GQA
            assert num_heads % num_groups == 0
            # 1. q keeps the full resolution
            self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
            # 2. k and v are reduced according to n_groups
            self.k_proj = nn.Parameter(
                torch.empty(self.d_head * self.num_groups, d_model)
            )
            self.v_proj = nn.Parameter(
                torch.empty(self.d_head * self.num_groups, d_model)
            )
            # 3. output proj remains the same
            self.output_proj = nn.Parameter(torch.empty(d_model, d_model))

        std = math.sqrt(1.0 / d_model)
        init.trunc_normal_(self.q_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.k_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.v_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)
        init.trunc_normal_(self.output_proj, mean=0.0, std=std, a=-3 * std, b=3 * std)

        # Init rope
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_head=self.d_head, max_seq_len=max_seq_len
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        if self.num_groups is None:
            return self._forward_mha(x, token_positions)
        else:
            return self._forward_gqa(x, token_positions)

    def _forward_mha(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        # x: (... seq_len d_model)

        # Reshape weights for multi-head
        # Weights are (num_heads * d_head, d_model), reshape to (num_heads, d_head, d_model)
        q_proj = rearrange(
            self.q_proj,
            "(num_heads d_head) d_model -> num_heads d_head d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )
        k_proj = rearrange(
            self.k_proj,
            "(num_heads d_head) d_model -> num_heads d_head d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )
        v_proj = rearrange(
            self.v_proj,
            "(num_heads d_head) d_model -> num_heads d_head d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )

        # Compute Q, K, V
        q = einsum(
            q_proj,
            x,
            "num_heads d_head d_model, ... seq_len d_model -> ... num_heads seq_len d_head",
        )
        k = einsum(
            k_proj,
            x,
            "num_heads d_head d_model, ... seq_len d_model -> ... num_heads seq_len d_head",
        )
        v = einsum(
            v_proj,
            x,
            "num_heads d_head d_model, ... seq_len d_model -> ... num_heads seq_len d_head",
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
        agg_v = scaled_dot_product(
            q=q,
            k=k,
            v=v,
            num_kv_groups=None,
            mask=mask,
        )
        agg_v = rearrange(agg_v, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        # Cast back to d_model
        output = einsum(
            # agg_v, self.output_proj, "... seq_len d_in, d_out d_in -> ... seq_len d_out"
            self.output_proj,
            agg_v,
            "d_out d_in, ... seq_len d_in -> ... seq_len d_out",
        )

        return output

    def _forward_gqa(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        # x: (... seq_len d_model)

        # Reshape weights for multi-head and multi-groups
        # q_proj -> (num_heads d_head d_model)
        # k_proj -> (num_groups d_head d_model)
        # v_proj -> (num_groups d_head d_model)
        q_proj = rearrange(
            self.q_proj,
            "(num_heads d_head) d_model -> num_heads d_head d_model",
            num_heads=self.num_heads,
            d_model=self.d_model,
        )
        k_proj = rearrange(
            self.k_proj,
            "(num_groups d_head) d_model -> num_groups d_head d_model",
            num_groups=self.num_groups,
            d_model=self.d_model,
        )
        v_proj = rearrange(
            self.v_proj,
            "(num_groups d_head) d_model -> num_groups d_head d_model",
            num_groups=self.num_groups,
            d_model=self.d_model,
        )

        # Compute Q, K, V
        q = einsum(
            q_proj,
            x,
            "num_heads d_head d_model, ... seq_len d_model -> ... num_heads seq_len d_head",
        )
        k = einsum(
            k_proj,
            x,
            "num_groups d_head d_model, ... seq_len d_model -> ... num_groups seq_len d_head",
        )
        v = einsum(
            v_proj,
            x,
            "num_groups d_head d_model, ... seq_len d_model -> ... num_groups seq_len d_head",
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
        agg_v = scaled_dot_product(
            q=q, k=k, v=v, num_kv_groups=self.num_groups, mask=mask
        )
        agg_v = rearrange(agg_v, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        # Cast back to d_model
        output = einsum(
            # agg_v, self.output_proj, "... seq_len d_in, d_out d_in -> ... seq_len d_out"
            self.output_proj,
            agg_v,
            "d_out d_in, ... seq_len d_in -> ... seq_len d_out",
        )

        return output
