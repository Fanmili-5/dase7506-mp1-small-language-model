"""Training helpers for a fixed train-only count/neural probability mixture."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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


def count_target_probability(
    counts: nn.Module, ids: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Query exact CSR count probability only at the supplied target IDs."""
    if ids.shape != targets.shape or ids.ndim != 2:
        raise ValueError("IDs and targets must have matching [batch,time] shapes")
    batch, length = ids.shape
    target = targets.clamp_min(0).flatten()
    result = counts.unigram[target].clone()
    positions = torch.arange(length, device=ids.device).expand(batch, -1)
    for order, table in enumerate(counts.tables, 2):
        history = order - 1
        if history > length or table.keys.numel() == 0:
            continue
        keys = torch.zeros_like(ids)
        for lag in range(history - 1, -1, -1):
            shifted = ids if lag == 0 else F.pad(ids[:, :-lag], (lag, 0))
            keys = keys * counts.vocab + shifted
        keys = keys.flatten()
        locations = torch.searchsorted(table.keys, keys).clamp_max(
            table.keys.numel() - 1
        )
        found = ((table.keys[locations] == keys)
                 & (positions.flatten() >= history - 1))
        result.mul_(torch.where(found, table.backoff[locations], 1.0))
        sizes = (table.offsets[locations + 1] - table.offsets[locations]) * found
        rows = torch.repeat_interleave(
            torch.arange(keys.numel(), device=ids.device), sizes
        )
        starts = torch.repeat_interleave(table.offsets[locations], sizes)
        local = (torch.arange(rows.numel(), device=ids.device)
                 - torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes))
        edges = starts + local
        matches = table.values[edges].long() == target[rows]
        increment = torch.zeros_like(result)
        increment.scatter_add_(0, rows[matches], table.mass[edges[matches]])
        result.add_(increment)
    return result.view_as(targets)


def mixture_target_log_probs(
    neural_logp: torch.Tensor,
    count_probability: torch.Tensor,
    targets: torch.Tensor,
    count_weight: float,
) -> torch.Tensor:
    neural_target = neural_logp.gather(
        -1, targets.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    return torch.logaddexp(
        neural_target.float() + math.log1p(-count_weight),
        count_probability.float().log() + math.log(count_weight),
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
