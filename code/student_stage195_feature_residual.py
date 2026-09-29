"""Frozen contextual base plus a small per-token nonlinear residual head.

The base Transformer already contextualizes each token causally. This adapter
does not repeat temporal attention, reducing CPU cost relative to Stage194.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from student_stage194_residual import base_log_probs_and_hidden


class FeatureResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(288)
        self.mlp = nn.Sequential(
            nn.Linear(288, 576), nn.GELU(), nn.Linear(576, 288),
        )
        self.low_rank = nn.Linear(288, 64)
        self.output_vocab = nn.Linear(64, 2048, bias=False)
        nn.init.zeros_(self.output_vocab.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != 288 or hidden.shape[1] > 256:
            raise ValueError("Expected independent causal windows [batch,time,288]")
        state = hidden + self.mlp(self.norm(hidden))
        return self.output_vocab(F.gelu(self.low_rank(state)))


class FeatureResidualGatedLM(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.expert = FeatureResidual()
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
        return F.log_softmax(base_logp + self.expert(hidden).float(), dim=-1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.predict_log_probs(ids)
