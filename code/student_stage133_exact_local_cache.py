"""Causal within-window successor cache over the Stage115 predictor."""
from __future__ import annotations

from collections import Counter, defaultdict

import torch

import student_stage115_order5_gate


MIN_ORDER = 2
MAX_ORDER = 8
CACHE_WEIGHT = 0.10


def add_exact_local_cache(distribution: torch.Tensor,
                          ids: torch.Tensor) -> torch.Tensor:
    """Mix fixed local successor distributions into caller-owned probabilities.

    A predecessor ending at position j contributes only its already observed
    successor at j+1. No target tensor, future token, or cross-row cache exists.
    """
    if (ids.ndim != 2 or distribution.ndim != 3
            or distribution.shape[:2] != ids.shape):
        raise ValueError("Expected [batch, time] IDs and [batch, time, vocab] probabilities")
    if not distribution.is_contiguous():
        raise ValueError("Expected contiguous caller-owned probabilities")
    batch, length = ids.shape
    vocab = distribution.shape[-1]
    if ids.numel() and (ids.min() < 0 or ids.max() >= vocab):
        raise ValueError("Token ID outside vocabulary")

    covered_rows: list[int] = []
    edge_rows: list[int] = []
    edge_tokens: list[int] = []
    edge_masses: list[float] = []
    for row, values in enumerate(ids.tolist()):
        tables = [None] + [defaultdict(Counter) for _ in range(MAX_ORDER)]
        for position in range(length):
            if position:
                predecessor = position - 1
                for order in range(1, min(MAX_ORDER, position) + 1):
                    key = tuple(values[predecessor - order + 1:predecessor + 1])
                    tables[order][key][values[position]] += 1
            for order in range(min(MAX_ORDER, position + 1), MIN_ORDER - 1, -1):
                key = tuple(values[position - order + 1:position + 1])
                counts = tables[order].get(key)
                if counts:
                    flat_row = row * length + position
                    covered_rows.append(flat_row)
                    total = sum(counts.values())
                    for token, count in counts.items():
                        edge_rows.append(flat_row)
                        edge_tokens.append(token)
                        edge_masses.append(CACHE_WEIGHT * count / total)
                    break

    if not covered_rows:
        return distribution
    flat = distribution.view(batch * length, vocab)
    device = distribution.device
    rows = torch.tensor(covered_rows, dtype=torch.long, device=device)
    flat[rows] *= 1 - CACHE_WEIGHT
    edges = torch.tensor(edge_rows, dtype=torch.long, device=device)
    tokens = torch.tensor(edge_tokens, dtype=torch.long, device=device)
    masses = torch.tensor(edge_masses, dtype=distribution.dtype, device=device)
    flat.index_put_((edges, tokens), masses, accumulate=True)
    return distribution


class ExactLocalCacheLM(student_stage115_order5_gate.Order5GatedLM):
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        base = super().forward(ids).exp()
        add_exact_local_cache(base, ids)
        return base.log()


def build_model(config: dict) -> ExactLocalCacheLM:
    return ExactLocalCacheLM(config)
