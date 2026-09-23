"""Training helpers for a fixed train-only count/neural probability mixture."""
from __future__ import annotations

import math

import torch
from torch import nn


def mixture_log_probs(
    neural_logp: torch.Tensor,
    count_logp: torch.Tensor,
    count_weight: float,
) -> torch.Tensor:
    if neural_logp.shape != count_logp.shape:
        raise ValueError("Experts must return the same distribution shape")
    if not 0 < count_weight < 1:
        raise ValueError("Training mixture requires both experts")
    return torch.logaddexp(
        neural_logp.float() + math.log1p(-count_weight),
        count_logp.float() + math.log(count_weight),
    )


def symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_probability = first.exp()
    second_probability = second.exp()
    return .5 * (
        (first_probability * (first - second)).sum(-1).mean()
        + (second_probability * (second - first)).sum(-1).mean()
    )


class FixedMixtureLM(nn.Module):
    def __init__(self, neural: nn.Module, counts: nn.Module, count_weight: float):
        super().__init__()
        self.neural = neural
        self.counts = counts
        self.count_weight = float(count_weight)
        self.context = neural.context
        self.vocab = neural.vocab

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return mixture_log_probs(
            self.neural.predict_log_probs(ids),
            self.counts.predict_log_probs(ids),
            self.count_weight,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.predict_log_probs(ids)
