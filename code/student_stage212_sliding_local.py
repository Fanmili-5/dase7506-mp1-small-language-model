"""Causal seven-token attention blocks used in the fixed Stage212 core."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import student


class SlidingLocalAttentionBlock(nn.Module):
    """Pre-norm local attention and the unchanged SwiGLU residual branch."""

    def __init__(self, config: dict):
        super().__init__()
        width = int(config["width"])
        heads = int(config["heads"])
        context = int(config["context"])
        window = int(config["local_attention_window"])
        if width % heads or not 2 <= window <= context:
            raise ValueError("Invalid Stage212 heads or local window")
        if str(config.get("position", "")).lower() != "rope":
            raise ValueError("Stage212 requires the fixed RoPE position scheme")
        self.heads = heads
        self.head_dim = width // heads
        self.window = window
        self.dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", True))
        norm = str(config.get("norm", "layernorm")).lower()
        eps = float(config.get("norm_eps", 1e-5))
        self.norm1 = student.make_norm(norm, width, eps)
        self.norm2 = student.make_norm(norm, width, eps)
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        hidden = max(1, int(round(width * float(config.get("mlp_ratio", 4.0)))))
        self.mlp = student.SwiGLU(width, hidden, bias)
        self.rope = student.RotaryEmbedding(
            self.head_dim, context, float(config.get("rope_base", 10_000.0)))
        position = torch.arange(context)[:, None]
        offsets = torch.arange(window - 1, -1, -1)[None, :]
        indices = position - offsets
        self.register_buffer("local_index", indices.clamp_min(0), persistent=False)
        self.register_buffer("local_valid", indices >= 0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = self.rope(q, k)
        indices = self.local_index[:length]
        local_k = k[:, :, indices, :]
        local_v = v[:, :, indices, :]
        scores = (q.unsqueeze(-2) * local_k).sum(-1).float()
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.local_valid[:length], -1.0e9)
        weights = F.softmax(scores, dim=-1).to(dtype=v.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = (weights.unsqueeze(-1) * local_v).sum(-2)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(attended), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


def install_local_attention_layers(model: nn.Module, config: dict) -> None:
    layers = tuple(int(value) for value in config.get("local_attention_layers", ()))
    if layers != (2, 4, 6, 8) or len(model.blocks) != 8:
        raise ValueError("Stage212 fixes four alternating local attention layers")
    for layer in layers:
        block = SlidingLocalAttentionBlock(config)
        block.apply(model._initialize)
        if bool(config.get("scaled_residual_init", False)):
            residual_std = 0.02 / math.sqrt(2 * len(model.blocks))
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.output.weight, std=residual_std)
        model.blocks[layer - 1] = block
    model.local_attention_layers = layers
