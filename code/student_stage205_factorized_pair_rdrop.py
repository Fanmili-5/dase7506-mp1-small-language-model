"""Shared low-rank causal adjacent-token interaction for the hybrid LM."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_rdrop


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


class FactorizedPairHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.pair_rank = int(config["pair_rank"])
        self.pair_scale = float(config["pair_scale"])
        if self.pair_rank != 64 or self.pair_scale != 0.75:
            raise ValueError("Unexpected fixed factorized-pair architecture")
        self.previous_pair = nn.Embedding(self.vocab, self.pair_rank)
        self.current_pair = nn.Embedding(self.vocab, self.pair_rank)
        self.pair_projection = nn.Linear(self.pair_rank, int(config["width"]), bias=False)
        nn.init.normal_(self.previous_pair.weight, std=0.1)
        nn.init.normal_(self.current_pair.weight, std=0.1)
        nn.init.normal_(self.pair_projection.weight, std=0.02)

    def pair_residual(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("ids must be independent [batch,time<=256] rows")
        previous = F.pad(ids[:, :-1], (1, 0), value=0)
        factors = self.previous_pair(previous) * self.current_pair(ids)
        pair = self.pair_projection(factors * self.pair_rank ** 0.5)
        return torch.cat((torch.zeros_like(pair[:, :1]), pair[:, 1:]), dim=1)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return super().input_embeddings(ids) + self.pair_scale * self.pair_residual(ids)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> FactorizedPairHybridLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Pinned hybrid parent source changed")
    return FactorizedPairHybridLM(config)
