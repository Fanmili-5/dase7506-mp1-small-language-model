"""Stage54 hybrid with a zero-start query-dependent gate after global SDPA."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_rdrop


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


class HeadwiseGatedAttentionBlock(nn.Module):
    """Reuse Stage54 initialized weights and gate the per-head SDPA output."""

    def __init__(self, original: student.Block, width: int):
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
        self.head_gate = nn.Linear(width, self.heads, bias=True)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        normalized = self.norm1(x)
        q, k, v = (
            self.qkv(normalized)
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        if self.rope is not None:
            q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        gate = (2.0 * torch.sigmoid(self.head_gate(normalized.float())))
        attended = attended * gate.transpose(1, 2).unsqueeze(-1).to(attended.dtype)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(attended), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class HeadwiseGatedHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        if config.get("attention_gate_kind") != "headwise_post_sdpa_zero_start":
            raise ValueError("Stage217 fixes headwise post-SDPA gating")
        super().__init__(config)
        width = int(config["width"])
        for index, block in enumerate(self.blocks):
            if isinstance(block, student.Block):
                self.blocks[index] = HeadwiseGatedAttentionBlock(block, width)


def build_model(config: dict) -> HeadwiseGatedHybridLM:
    if hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 R-Drop source changed")
    return HeadwiseGatedHybridLM(config)
