"""Causal hashed adjacent-token input representation for the hybrid LM."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_rdrop


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
FIRST_MULTIPLIER = 1_315_423_911
SECOND_MULTIPLIER = 2_654_435_761


class HashedBigramHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.bigram_buckets = int(config["bigram_buckets"])
        self.bigram_scale = float(config["bigram_scale"])
        if self.bigram_buckets != 8192 or not 0 <= self.bigram_scale <= 1:
            raise ValueError("Unexpected hashed-bigram architecture")
        self.bigram = nn.Embedding(self.bigram_buckets + 1, int(config["width"]),
                                   padding_idx=0)
        nn.init.normal_(self.bigram.weight, std=0.02)
        with torch.no_grad():
            self.bigram.weight[0].zero_()

    def bigram_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("ids must be independent [batch,time<=256] rows")
        previous = F.pad(ids[:, :-1], (1, 0), value=0)
        buckets = torch.remainder(previous * FIRST_MULTIPLIER
                                  + ids * SECOND_MULTIPLIER,
                                  self.bigram_buckets) + 1
        return torch.cat((torch.zeros_like(buckets[:, :1]), buckets[:, 1:]), dim=1)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        token = super().input_embeddings(ids)
        return token + self.bigram_scale * self.bigram(self.bigram_ids(ids))

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        # Parent inference bypasses input_embeddings, so keep train/eval inputs
        # identical here. No state crosses independent evaluator rows.
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


def build_model(config: dict) -> HashedBigramHybridLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Pinned hybrid parent source changed")
    return HashedBigramHybridLM(config)
