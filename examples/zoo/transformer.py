from typing import Any, Callable, Literal

import einops
import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Replace with default pytorch implementations


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-4):
        super().__init__()
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        x = x - x.mean(axis=-1)[..., None]
        x = x / (x.std(axis=-1)[..., None] + self.epsilon)
        x = x * self.w_ln
        x = x + self.b_ln
        return x


class Attention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_model, num_heads, d_head, num_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones((num_ctx, num_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
            1 - self.mask[: x.shape[-2], : x.shape[-2]]
        )
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: Callable = F.relu):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_fn = act_fn
        # self.ln = LayerNorm(d_mlp)

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = self.act_fn(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu,
    ):
        super().__init__()
        # self.ln1 = LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, d_head, num_ctx)
        # self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, act_fn)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp((x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu,
        use_ln: bool = True,
    ):
        super().__init__()

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(num_ctx, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_mlp, d_head, num_heads, num_ctx, act_fn)
                for i in range(num_layers)
            ]
        )
        # self.ln = LayerNorm(d_model)
        self.unembed = Unembed(d_vocab, d_model)
        # self.use_ln = use_ln

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x


ActivationFunctionLiteral = Literal["relu", "gelu", "silu", "tanh"]


class TransformerConfig(pydantic.BaseModel):
    d_vocab: int
    num_layers: int = 1
    d_model: int = 128
    d_mlp: int = 128 * 4
    d_head: int = 128 // 4
    num_heads: int = 4
    num_ctx: int = 3
    act_fn: ActivationFunctionLiteral = "relu"

    def factory(self):
        act_fn = getattr(F, self.act_fn)

        return Transformer(
            num_layers=self.num_layers,
            d_vocab=self.d_vocab,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            d_head=self.d_head,
            num_heads=self.num_heads,
            num_ctx=self.num_ctx,
            act_fn=act_fn,
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def determine_mlp_and_head(cls, data: Any):
        defaults = {"num_layers": 1, "d_model": 128, "num_heads": 4, "num_ctx": 3}
        defaults.update(data)

        if not data.get("d_mlp"):
            data["d_mlp"] = defaults["d_model"] * 4
        if not data.get("d_head"):
            data["d_head"] = defaults["d_model"] // defaults["num_heads"]

        return data
