"""Zero-start parallel causal attention inside the existing gated conv blocks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_structured


class ParallelGlobalLocalBlock(student_hybrid_conv_structured.GatedCausalConvBlock):
    """Retain the local path and add a trainable global path at one depth."""

    def __init__(self, config: dict):
        super().__init__(config)
        width = int(config["width"])
        heads = int(config["heads"])
        if width % heads or (width // heads) % 2:
            raise ValueError("Parallel RoPE attention needs an even head dimension")
        if str(config.get("position", "")).lower() != "rope":
            raise ValueError("Stage177 fixes RoPE for its global branch")
        self.heads = heads
        self.head_dim = width // heads
        bias = bool(config.get("bias", True))
        self.attn_qkv = nn.Linear(width, 3 * width, bias=bias)
        self.attn_proj = nn.Linear(width, width, bias=bias)
        self.attn_rope = student.RotaryEmbedding(
            self.head_dim, int(config["context"]), float(config.get("rope_base", 10_000.0)))
        nn.init.normal_(self.attn_qkv.weight, std=.02)
        nn.init.zeros_(self.attn_proj.weight)
        if self.attn_qkv.bias is not None:
            nn.init.zeros_(self.attn_qkv.bias)
        if self.attn_proj.bias is not None:
            nn.init.zeros_(self.attn_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        normalized = self.norm1(x)
        gate, value = self.input(normalized).chunk(2, dim=-1)
        value = F.pad(value.transpose(1, 2), (self.kernel - 1, 0))
        value = self.depthwise(value).transpose(1, 2)
        local = self.proj(F.silu(gate) * value)

        q, k, v = (self.attn_qkv(normalized)
                   .view(batch, length, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))
        q, k = self.attn_rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                   is_causal=True)
        global_path = self.attn_proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + F.dropout(local + global_path, self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


def install_parallel_branches(model: nn.Module, config: dict) -> None:
    layers = tuple(int(value) for value in config["conv_layers"])
    if layers != (2, 4, 6, 8) or len(model.blocks) != 8:
        raise ValueError("Stage177 fixes the four Stage54 local blocks")
    # Initializing extra branch parameters must not change the dropout stream
    # relative to a Stage54 model constructed from the same seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(177017)
        for layer in layers:
            previous = model.blocks[layer - 1]
            if not isinstance(previous, student_hybrid_conv_structured.GatedCausalConvBlock):
                raise ValueError("Expected a Stage54 gated causal-convolution block")
            replacement = ParallelGlobalLocalBlock(config)
            outcome = replacement.load_state_dict(previous.state_dict(), strict=False)
            if outcome.unexpected_keys or set(outcome.missing_keys) != {
                    "attn_qkv.weight", "attn_proj.weight"}:
                raise ValueError("Original Stage54 block parameters did not transfer exactly")
            model.blocks[layer - 1] = replacement
