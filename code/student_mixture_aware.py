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
    counts: nn.Module, ids: torch.Tensor, targets: torch.Tensor,
    edge_keys: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Query exact CSR count probability only at the supplied target IDs."""
    if ids.shape != targets.shape or ids.ndim != 2:
        raise ValueError("IDs and targets must have matching [batch,time] shapes")
    batch, length = ids.shape
    if edge_keys is None:
        edge_keys = build_target_edge_keys(counts)
    if len(edge_keys) != len(counts.tables):
        raise ValueError("Expected one target lookup per count order")
    target = targets.clamp_min(0).flatten()
    result = counts.unigram[target].clone()
    positions = torch.arange(length, device=ids.device).expand(batch, -1)
    for order, (table, table_edge_keys) in enumerate(
        zip(counts.tables, edge_keys), 2
    ):
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
        # Use the compact CSR row number rather than the raw base-vocabulary
        # context code.  A five-token context fits int64, but appending its
        # target would require 66 bits for vocab 2048.
        queries = locations * counts.vocab + target
        edge_locations = torch.searchsorted(
            table_edge_keys, queries
        ).clamp_max(table_edge_keys.numel() - 1)
        edge_found = found & (table_edge_keys[edge_locations] == queries)
        result.add_(torch.where(edge_found, table.mass[edge_locations], 0.0))
    return result.view_as(targets)


def build_target_edge_keys(counts: nn.Module) -> tuple[torch.Tensor, ...]:
    """Build sorted overflow-safe (CSR-row,target) keys once per count model."""
    result = []
    for table in counts.tables:
        sizes = table.offsets[1:] - table.offsets[:-1]
        rows = torch.repeat_interleave(
            torch.arange(table.keys.numel(), device=table.keys.device), sizes
        )
        keys = rows * counts.vocab + table.values.long()
        if keys.numel() != table.mass.numel() or (
                keys.numel() > 1 and (keys[1:] <= keys[:-1]).any()):
            raise ValueError("Count edges must be unique and lexicographically sorted")
        result.append(keys)
    return tuple(result)


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
