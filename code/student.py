"""Configurable causal language model used for MP1 experiments.

The defaults reproduce the baseline design closely. Individual mechanisms can
be enabled from JSON configs so every claimed change has a clean ablation. The
module is self-contained because the fixed evaluator records its SHA-256.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    """Precomputed rotary-position factors for one attention head."""

    def __init__(self, head_dim: int, context: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        positions = torch.arange(context, dtype=torch.float32)
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[-2]
        cos = self.cos[:, :, :length].to(dtype=q.dtype)
        sin = self.sin[:, :, :length].to(dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, bias: bool):
        super().__init__()
        self.input = nn.Linear(width, 2 * hidden, bias=bias)
        self.output = nn.Linear(hidden, width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


def make_norm(kind: str, width: int, eps: float) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(width, eps=eps)
    if kind == "rmsnorm":
        return nn.RMSNorm(width, eps=eps)
    raise ValueError(f"Unknown norm: {kind!r}")


class Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["width"])
        heads = int(config["heads"])
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", True))
        norm = str(config.get("norm", "layernorm")).lower()
        eps = float(config.get("norm_eps", 1e-5))
        self.norm1 = make_norm(norm, width, eps)
        self.norm2 = make_norm(norm, width, eps)
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)

        activation = str(config.get("activation", "gelu")).lower()
        hidden = max(1, int(round(width * float(config.get("mlp_ratio", 4.0)))))
        if activation == "gelu":
            self.mlp = nn.Sequential(
                nn.Linear(width, hidden, bias=bias),
                nn.GELU(),
                nn.Linear(hidden, width, bias=bias),
            )
        elif activation == "swiglu":
            self.mlp = SwiGLU(width, hidden, bias)
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

        position = str(config.get("position", "learned")).lower()
        self.rope = (
            RotaryEmbedding(self.head_dim, int(config["context"]), float(config.get("rope_base", 10_000.0)))
            if position == "rope"
            else None
        )
        if position not in {"learned", "rope"}:
            raise ValueError(f"Unknown position encoding: {position!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        if self.rope is not None:
            q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(attended), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class StudentLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.context = int(config["context"])
        self.vocab = int(config["vocab"])
        width = int(config["width"])
        depth = int(config["depth"])
        if self.context != 256 or self.vocab != 2048:
            raise ValueError("MP1 fixes context=256 and vocab=2048.")

        self.token = nn.Embedding(self.vocab, width)
        self.position_kind = str(config.get("position", "learned")).lower()
        self.pos = nn.Embedding(self.context, width) if self.position_kind == "learned" else None
        self.embedding_dropout = float(config.get("dropout", 0.0))
        self.blocks = nn.ModuleList([Block(config) for _ in range(depth)])
        self.norm = make_norm(
            str(config.get("norm", "layernorm")).lower(),
            width,
            float(config.get("norm_eps", 1e-5)),
        )
        self.head = nn.Linear(width, self.vocab, bias=False)

        self.apply(self._initialize)
        if bool(config.get("scaled_residual_init", False)):
            residual_std = 0.02 / math.sqrt(2 * depth)
            for block in self.blocks:
                nn.init.normal_(block.proj.weight, std=residual_std)
                mlp_output = block.mlp.output if isinstance(block.mlp, SwiGLU) else block.mlp[-1]
                nn.init.normal_(mlp_output.weight, std=residual_std)
        self.head.weight = self.token.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] > self.context:
            raise ValueError(f"ids must have shape [batch, time<={self.context}]")
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        """Return finite, normalized, causal natural-log probabilities."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config: dict) -> StudentLM:
    return StudentLM(config)
