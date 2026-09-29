"""Stage54 R-Drop Transformer with fixed train-derived semantic input rows."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_rdrop


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


class SemanticInputHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        if int(config.get("semantic_dim", 0)) != 64:
            raise ValueError("Stage215 fixes semantic_dim=64")
        super().__init__(config)
        self.register_buffer("semantic_basis", torch.zeros(self.vocab, 64))
        self.semantic_projection = nn.Linear(64, int(config["width"]), bias=False)
        nn.init.zeros_(self.semantic_projection.weight)

    def set_semantic_basis(self, basis: torch.Tensor) -> None:
        if basis.shape != self.semantic_basis.shape or not torch.isfinite(basis).all():
            raise ValueError("Invalid fixed train-only semantic basis")
        self.semantic_basis.copy_(basis.to(self.semantic_basis.device))

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        token = super().input_embeddings(ids)
        semantic = F.embedding(ids, self.semantic_basis)
        return token + self.semantic_projection(semantic)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        # RegularizedLM's evaluation shortcut bypasses input_embeddings;
        # this explicit path keeps the train-derived stream in both modes.
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent [batch,time<=256] rows")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config: dict) -> SemanticInputHybridLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Stage54 R-Drop source changed")
    return SemanticInputHybridLM(config)
