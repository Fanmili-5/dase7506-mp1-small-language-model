"""Causal train-derived input-pair representation shared by train and export."""
from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


VOCAB = 2048
TOP_K = 16384
DIM = 16


def build_train_bigram_rank_map(tokens: torch.Tensor) -> torch.Tensor:
    """Rank the top pairs from supplied *training* IDs only, deterministically."""
    if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.device.type != "cpu":
        raise ValueError("Expected a one-dimensional CPU training-token stream")
    if len(tokens) < 2 or int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB:
        raise ValueError("Invalid training token IDs")
    pairs = (tokens[:-1] * VOCAB + tokens[1:]).tolist()
    counts = Counter(pairs)
    selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_K]
    if len(selected) != TOP_K:
        raise ValueError("Not enough distinct training bigrams")
    mapping = torch.zeros(VOCAB * VOCAB, dtype=torch.int16)
    for rank, (pair, _) in enumerate(selected, 1):
        mapping[pair] = rank
    return mapping


class BigramInputMixin:
    """The extra input at t reads x[t-1],x[t] and no other row/window."""

    def __init__(self, config: dict):
        if (int(config.get("bigram_top_k", -1)) != TOP_K
                or int(config.get("bigram_dim", -1)) != DIM
                or int(config.get("vocab", -1)) != VOCAB):
            raise ValueError("Fixed Stage223 vocabulary, table size and rank")
        super().__init__(config)
        self.register_buffer("bigram_rank_map",
                             torch.zeros(VOCAB * VOCAB, dtype=torch.int16))
        self.bigram_embedding = nn.Embedding(TOP_K + 1, DIM, padding_idx=0)
        self.bigram_projection = nn.Linear(DIM, int(config["width"]), bias=False)
        nn.init.normal_(self.bigram_embedding.weight, std=0.02)
        with torch.no_grad():
            self.bigram_embedding.weight[0].zero_()
        nn.init.normal_(self.bigram_projection.weight, std=0.02)

    def set_bigram_rank_map(self, mapping: torch.Tensor) -> None:
        if (mapping.shape != self.bigram_rank_map.shape
                or mapping.dtype != torch.int16 or mapping.device.type != "cpu"
                or int(mapping.min()) != 0 or int(mapping.max()) != TOP_K
                or int((mapping > 0).sum()) != TOP_K):
            raise ValueError("Invalid train-derived bigram rank map")
        self.bigram_rank_map.copy_(mapping.to(self.bigram_rank_map.device))

    def pair_ranks(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent [batch,time<=256] rows")
        previous = F.pad(ids[:, :-1], (1, 0), value=0)
        keys = previous * VOCAB + ids
        ranks = self.bigram_rank_map[keys].long()
        # An independent evaluation window has no token before x[0].
        return torch.cat((torch.zeros_like(ranks[:, :1]), ranks[:, 1:]), dim=1)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        parent = super()
        base = (parent.input_embeddings(ids)
                if hasattr(parent, "input_embeddings") else self.token(ids))
        pair = self.bigram_projection(self.bigram_embedding(self.pair_ranks(ids)))
        return base + pair

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent [batch,time<=256] rows")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
