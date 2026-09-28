"""Stage54 hybrid with a learned causal distance bias in global attention."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_rdrop


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


def distance_buckets(context: int, exact: int = 16, count: int = 32) -> torch.Tensor:
    if context != 256 or exact != 16 or count != 32:
        raise ValueError("Stage216 fixes 256 positions and 16+16 causal buckets")
    positions = torch.arange(context)
    distance = (positions[:, None] - positions[None, :]).clamp_min(0)
    logarithmic = exact + torch.floor(
        torch.log(distance.clamp_min(exact).float() / exact)
        / math.log(context / exact) * (count - exact)
    ).long()
    return torch.where(distance < exact, distance, logarithmic).clamp_max(count - 1)


class RelativeBiasAttentionBlock(nn.Module):
    """Reuses initialized Stage54 weights; zero bias is its original attention."""

    def __init__(self, original: student.Block, context: int):
        super().__init__()
        self.heads = original.heads
        self.head_dim = original.head_dim
        self.dropout = original.dropout
        self.norm1 = original.norm1
        self.norm2 = original.norm2
        self.qkv = original.qkv
        self.proj = original.proj
        self.mlp = original.mlp
        self.rope = original.rope
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, 32))
        self.register_buffer("bucket_indices", distance_buckets(context), persistent=False)
        future = torch.ones(context, context, dtype=torch.bool).triu(1)
        self.register_buffer("future_mask", future, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        if self.rope is not None:
            q, k = self.rope(q, k)
        bias = self.relative_bias[:, self.bucket_indices[:length, :length]].to(q.dtype)
        bias = bias.masked_fill(self.future_mask[:length, :length], float("-inf"))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias[None],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(attended), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class RelativeBiasHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        if int(config.get("relative_bias_buckets", 0)) != 32:
            raise ValueError("Stage216 fixes 32 relative-distance buckets")
        super().__init__(config)
        for index, block in enumerate(self.blocks):
            if isinstance(block, student.Block):
                self.blocks[index] = RelativeBiasAttentionBlock(block, self.context)


def build_model(config: dict) -> RelativeBiasHybridLM:
    if hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 R-Drop source changed")
    return RelativeBiasHybridLM(config)
