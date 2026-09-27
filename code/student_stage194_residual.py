"""Small causal residual expert above one frozen Stage105/Stage143 predictor.

The input-only adapter never sees the next-token target or cross-window state.
Its zero-initialized output preserves the base distribution at initialization.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def base_log_probs_and_hidden(base: nn.Module,
                              ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass algebraic copy of the Stage105 inference path."""
    if ids.ndim != 2 or not 1 <= ids.shape[1] <= 256:
        raise ValueError("Expected independent causal windows of at most 256 tokens")
    hidden = base.neural.features(ids)
    with torch.autocast(device_type=ids.device.type, enabled=False):
        hidden = hidden.float()
        result = F.softmax(
            (base.neural.head(hidden) + base.neural.output_bias)
            / base.temperature + base.prior_weight * base.log_prior,
            dim=-1,
        )
        gate = base.neural.copy_gate(hidden) + base.copy_shift
        vocabulary_scale = torch.sigmoid(-gate)
        result.mul_(vocabulary_scale)
        query = base.neural.copy_query(hidden).float()
        key = base.neural.copy_key(hidden).float()
        scores = (query @ key.transpose(-1, -2)) * base.neural.copy_scale
        length = ids.shape[1]
        attention = scores.masked_fill(
            base.copy_future_mask[:length, :length], float("-inf")
        ).softmax(-1)
        attention.mul_(1 - vocabulary_scale)
        result.scatter_add_(
            -1, ids[:, None, :].expand(-1, length, -1), attention
        )
        top = result.topk(2, dim=-1).values.log()
        terms, suffix, backoff, max_mass = base.ngram.collect(ids)
        features = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                                backoff, max_mass), dim=-1).double()
        slope = ((features - base.gate_mean) / base.gate_std)
        slope = (slope * base.gate_coeff).sum(-1)
        weight = torch.sigmoid(base.anchor_logit + base.slope_scale * slope)
        weight = (1e-4 + (1 - 2e-4) * weight).float()
        result.mul_((1 - weight).unsqueeze(-1))
        base.ngram.add_collected(result, weight, terms, suffix)
        return result.log(), hidden


class ResidualExpert(nn.Module):
    """One causal attention/MLP block and a low-rank vocabulary correction."""

    def __init__(self, width: int = 288, vocab: int = 2048, rank: int = 64):
        super().__init__()
        if (width, vocab, rank) != (288, 2048, 64):
            raise ValueError("Unexpected Stage194 residual shape")
        self.input_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, 8, dropout=0.0, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_low_rank = nn.Linear(width, rank)
        self.output_vocab = nn.Linear(rank, vocab, bias=False)
        nn.init.zeros_(self.output_vocab.weight)
        self.register_buffer(
            "future_mask", torch.triu(torch.ones(256, 256, dtype=torch.bool),
                                      diagonal=1), persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != 288 or hidden.shape[1] > 256:
            raise ValueError("Expected causal [batch,time,288] hidden states")
        length = hidden.shape[1]
        normalized = self.input_norm(hidden)
        correction, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=self.future_mask[:length, :length],
            need_weights=False,
        )
        state = hidden + correction
        state = state + self.mlp(self.mlp_norm(state))
        low_rank = F.gelu(self.output_low_rank(self.output_norm(state)))
        return self.output_vocab(low_rank)


class ResidualGatedLM(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.expert = ResidualExpert()
        self.context, self.vocab = 256, 2048
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_logp, hidden = base_log_probs_and_hidden(self.base, ids)
        correction = self.expert(hidden)
        return F.log_softmax(base_logp + correction.float(), dim=-1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.predict_log_probs(ids)
