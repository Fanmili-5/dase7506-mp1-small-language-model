"""Alternating global-attention and gated causal-convolution Transformer."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_structured

PARENT_SHA = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class GatedCausalConvBlock(nn.Module):
    """Pre-norm causal local mixer followed by the ordinary SwiGLU FFN."""

    def __init__(self, config: dict):
        super().__init__()
        width = int(config["width"])
        kernel = int(config["conv_kernel"])
        if kernel < 2:
            raise ValueError("conv_kernel must be at least two")
        self.kernel = kernel
        self.dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", True))
        norm = str(config.get("norm", "layernorm")).lower()
        eps = float(config.get("norm_eps", 1e-5))
        self.norm1 = student.make_norm(norm, width, eps)
        self.norm2 = student.make_norm(norm, width, eps)
        self.input = nn.Linear(width, 2 * width, bias=bias)
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        hidden = max(1, int(round(width * float(config.get("mlp_ratio", 4.0)))))
        self.mlp = student.SwiGLU(width, hidden, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        gate, value = self.input(normalized).chunk(2, dim=-1)
        value = F.pad(value.transpose(1, 2), (self.kernel - 1, 0))
        value = self.depthwise(value).transpose(1, 2)
        mixed = F.silu(gate) * value
        x = x + F.dropout(self.proj(mixed), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


def install_conv_layers(model: nn.Module, config: dict) -> None:
    layers = tuple(int(value) for value in config.get("conv_layers", ()))
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("conv_layers must be unique and increasing")
    if layers[0] < 1 or layers[-1] > len(model.blocks):
        raise ValueError("conv_layers contains an invalid one-based layer index")
    depth = len(model.blocks)
    for layer in layers:
        block = GatedCausalConvBlock(config)
        block.apply(model._initialize)
        nn.init.normal_(block.depthwise.weight, std=0.02)
        if block.depthwise.bias is not None:
            nn.init.zeros_(block.depthwise.bias)
        if bool(config.get("scaled_residual_init", False)):
            residual_std = 0.02 / math.sqrt(2 * depth)
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.output.weight, std=residual_std)
        model.blocks[layer - 1] = block
    model.conv_layers = layers


class HybridConvStructuredLM(student_structured.StructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_conv_layers(self, config)


def build_model(config: dict) -> HybridConvStructuredLM:
    actual = hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Structured parent source changed")
    return HybridConvStructuredLM(config)
