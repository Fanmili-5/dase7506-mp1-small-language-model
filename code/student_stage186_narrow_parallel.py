"""Zero-start, quarter-width global attention beside Stage54 local blocks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_structured


class NarrowParallelBlock(student_hybrid_conv_structured.GatedCausalConvBlock):
    def __init__(self, config: dict):
        super().__init__(config)
        width = int(config["width"])
        heads = int(config["parallel_heads"])
        head_dim = int(config["parallel_head_dim"])
        if (width, heads, head_dim) != (288, 2, 36):
            raise ValueError("Stage186 fixes a two-head, 72-channel branch")
        if str(config.get("position", "")).lower() != "rope":
            raise ValueError("Stage186 fixes RoPE for the global branch")
        self.parallel_heads = heads
        self.parallel_head_dim = head_dim
        self.parallel_width = heads * head_dim
        bias = bool(config.get("bias", True))
        self.attn_qkv = nn.Linear(width, 3 * self.parallel_width, bias=bias)
        self.attn_proj = nn.Linear(self.parallel_width, width, bias=bias)
        self.attn_rope = student.RotaryEmbedding(
            head_dim, int(config["context"]), float(config.get("rope_base", 10_000.0)))
        nn.init.normal_(self.attn_qkv.weight, std=.02)
        nn.init.zeros_(self.attn_proj.weight)
        if self.attn_qkv.bias is not None:
            nn.init.zeros_(self.attn_qkv.bias)
        if self.attn_proj.bias is not None:
            nn.init.zeros_(self.attn_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        normalized = self.norm1(x)
        gate, value = self.input(normalized).chunk(2, dim=-1)
        value = F.pad(value.transpose(1, 2), (self.kernel - 1, 0))
        value = self.depthwise(value).transpose(1, 2)
        local = self.proj(F.silu(gate) * value)

        q, k, v = (self.attn_qkv(normalized)
                   .view(batch, length, 3, self.parallel_heads, self.parallel_head_dim)
                   .permute(2, 0, 3, 1, 4))
        q, k = self.attn_rope(q, k)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True)
        global_path = self.attn_proj(
            attended.transpose(1, 2).reshape(batch, length, self.parallel_width))
        x = x + F.dropout(local + global_path, self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


def install_narrow_branches(model: nn.Module, config: dict) -> None:
    layers = tuple(int(value) for value in config["conv_layers"])
    if layers != (2, 4, 6, 8) or len(model.blocks) != 8:
        raise ValueError("Stage186 fixes the four Stage54 local blocks")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(186017)
        for layer in layers:
            previous = model.blocks[layer - 1]
            if not isinstance(previous, student_hybrid_conv_structured.GatedCausalConvBlock):
                raise ValueError("Expected a Stage54 gated causal-convolution block")
            replacement = NarrowParallelBlock(config)
            outcome = replacement.load_state_dict(previous.state_dict(), strict=False)
            if outcome.unexpected_keys or set(outcome.missing_keys) != {
                    "attn_qkv.weight", "attn_proj.weight"}:
                raise ValueError("Original Stage54 block parameters did not transfer exactly")
            model.blocks[layer - 1] = replacement
