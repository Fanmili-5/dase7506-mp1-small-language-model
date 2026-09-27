"""Training-only hybrid with two causal low-rank linear-memory mixers.

The selected blocks replace ordinary softmax attention at layers 3 and 7.
Each replacement computes a within-window prefix sum of key/value outer
products. No state is carried between independent evaluator windows.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_rdrop


class LinearMemoryBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["width"])
        heads = int(config["heads"])
        key_dim = int(config["linear_memory_key_dim"])
        if width % heads or key_dim <= 0 or key_dim % 2:
            raise ValueError("width/heads and even linear-memory key dimension required")
        self.heads = heads
        self.value_dim = width // heads
        self.key_dim = key_dim
        self.dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", True))
        kind = str(config.get("norm", "layernorm")).lower()
        eps = float(config.get("norm_eps", 1e-5))
        self.norm1 = student.make_norm(kind, width, eps)
        self.norm2 = student.make_norm(kind, width, eps)
        self.qk = nn.Linear(width, 2 * heads * key_dim, bias=bias)
        self.value = nn.Linear(width, width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        self.rope = student.RotaryEmbedding(
            key_dim, int(config["context"]), float(config.get("rope_base", 10000.0)))
        hidden = max(1, int(round(width * float(config.get("mlp_ratio", 4.0)))))
        self.mlp = student.SwiGLU(width, hidden, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        normalized = self.norm1(x)
        q, k = self.qk(normalized).view(
            batch, length, 2, self.heads, self.key_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        # Positive features make the prefix-sum denominator nonnegative.
        q = F.elu(q.float()) + 1.0
        k = F.elu(k.float()) + 1.0
        v = self.value(normalized).reshape(
            batch, length, self.heads, self.value_dim).transpose(1, 2).float()
        if self.training and self.dropout:
            v = F.dropout(v, self.dropout, True)
        prefix_k = k.cumsum(dim=2)
        prefix_kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2)
        numerator = torch.matmul(q.unsqueeze(-2), prefix_kv).squeeze(-2)
        denominator = (q * prefix_k).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed = (numerator / denominator).transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(mixed.to(x.dtype)), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class LinearMemoryRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        layers = tuple(int(layer) for layer in config["linear_memory_layers"])
        if layers != (3, 7):
            raise ValueError("Stage206 fixes linear-memory layers 3 and 7")
        if set(layers) & set(self.conv_layers):
            raise ValueError("Linear-memory layers overlap convolution layers")
        depth = len(self.blocks)
        for layer in layers:
            block = LinearMemoryBlock(config)
            block.apply(self._initialize)
            if bool(config.get("scaled_residual_init", False)):
                residual_std = 0.02 / math.sqrt(2 * depth)
                nn.init.normal_(block.proj.weight, std=residual_std)
                nn.init.normal_(block.mlp.output.weight, std=residual_std)
            self.blocks[layer - 1] = block
        self.linear_memory_layers = layers


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> LinearMemoryRDropLM:
    return LinearMemoryRDropLM(config)
