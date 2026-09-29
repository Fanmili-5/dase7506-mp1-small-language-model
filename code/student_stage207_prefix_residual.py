"""Causal ASCII current-word prefix residual over the frozen Stage143 model.

The lexical features are derived only from the supplied tokenizer's input IDs.
They reset independently for each prediction row and never inspect targets.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from student_stage160_morphology import (
    HIDDEN_WIDTH, RESIDUAL_WIDTH, MorphologyResidual, TOKENIZER,
    TOKENIZER_SHA256, VOCAB,
)


MAX_PREFIX = 32
FEATURE_LETTERS = 16
CHAR_WIDTH = 32
CHAR_COUNT = 53
_START = re.compile(r"Ġ([A-Za-z]+)\Z")
_CONTINUATION = re.compile(r"[A-Za-z]+\Z")


def _token_pieces() -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    if hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Fixed tokenizer changed")
    vocab = json.loads(TOKENIZER.read_text(encoding="utf-8"))["model"]["vocab"]
    if len(vocab) != VOCAB or set(vocab.values()) != set(range(VOCAB)):
        raise ValueError("Unexpected course token vocabulary")
    starts: list[str | None] = [None] * VOCAB
    continuations: list[str | None] = [None] * VOCAB
    for spelling, token_id in vocab.items():
        match = _START.fullmatch(spelling)
        if match is not None:
            starts[token_id] = match.group(1)
        elif _CONTINUATION.fullmatch(spelling) is not None:
            continuations[token_id] = spelling
    return tuple(starts), tuple(continuations)


STARTS, CONTINUATIONS = _token_pieces()


def _char_id(letter: str) -> int:
    code = ord(letter)
    if 65 <= code <= 90:
        return code - 64
    if 97 <= code <= 122:
        return code - 70
    raise ValueError("Non-ASCII word-prefix character")


def prefix_features(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,T,16] character IDs and [B,T] active mask from input IDs."""
    if ids.ndim != 2 or ids.shape[1] > 256:
        raise ValueError("Expected independent [batch,time<=256] input rows")
    rows = ids.detach().to("cpu", dtype=torch.long).tolist()
    batch, length = len(rows), len(rows[0]) if rows else 0
    chars = torch.zeros((batch, length, FEATURE_LETTERS), dtype=torch.long)
    active = torch.zeros((batch, length), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        prefix: str | None = None
        for position, token_id in enumerate(row):
            if not 0 <= token_id < VOCAB:
                raise ValueError("Unexpected token ID")
            start = STARTS[token_id]
            continuation = CONTINUATIONS[token_id]
            if start is not None:
                prefix = start
            elif prefix is not None and continuation is not None:
                prefix += continuation
            else:
                prefix = None
            if prefix is not None and len(prefix) > MAX_PREFIX:
                prefix = None
            if prefix is not None:
                active[row_index, position] = 1.0
                suffix = prefix[-FEATURE_LETTERS:]
                chars[row_index, position, -len(suffix):] = torch.tensor(
                    [_char_id(letter) for letter in suffix], dtype=torch.long)
    return chars.to(ids.device), active.to(ids.device)


class PrefixConditionedResidual(MorphologyResidual):
    """Zero-start rank-128 output correction with an explicit observed prefix."""

    def __init__(self):
        super().__init__()
        self.char_embeddings = nn.Embedding(CHAR_COUNT, CHAR_WIDTH, padding_idx=0)
        nn.init.normal_(self.char_embeddings.weight, std=0.02)
        with torch.no_grad():
            self.char_embeddings.weight[0].zero_()
        self.prefix_input = nn.Sequential(
            nn.Linear(FEATURE_LETTERS * CHAR_WIDTH, RESIDUAL_WIDTH),
            nn.SiLU(),
            nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH),
        )
        self.residual_dropout = 0.25

    def forward(
        self,
        base_logp: torch.Tensor,
        hidden: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        if (base_logp.ndim != 3 or base_logp.shape[-1] != VOCAB
                or hidden.shape != (*base_logp.shape[:2], HIDDEN_WIDTH)
                or ids.shape != base_logp.shape[:2]
                or base_logp.device != hidden.device or ids.device != hidden.device):
            raise ValueError("Mismatched causal hidden, base rows or input IDs")
        chars, active = prefix_features(ids)
        spelling = self.char_embeddings(chars).flatten(-2)
        lexical = self.prefix_input(spelling)
        context = F.silu(self.input(self.pre(hidden.float())) + lexical)
        context = F.dropout(context, self.residual_dropout, self.training)
        context = self.output(context)
        delta = context @ self.token_rows().T / math.sqrt(RESIDUAL_WIDTH)
        delta = delta * active[..., None]
        return F.log_softmax(base_logp.float() + delta.float(), dim=-1)


class FrozenBaseWithPrefix(nn.Module):
    """Training/diagnostic view; not a resource-qualified CPU submission."""

    def __init__(self, base: nn.Module, residual: PrefixConditionedResidual):
        super().__init__()
        self.base = base.eval()
        self.residual = residual
        self.context = base.context
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_logp = self.base.predict_log_probs(ids).float()
            hidden = self.base.neural.features(ids).float()
        return self.residual(base_logp, hidden, ids)


def build_model(config: dict) -> PrefixConditionedResidual:
    expected = {"width": HIDDEN_WIDTH, "residual_width": RESIDUAL_WIDTH,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "prefix_letters": FEATURE_LETTERS, "char_width": CHAR_WIDTH}
    if config != expected:
        raise ValueError("Unexpected Stage207 prefix-residual configuration")
    return PrefixConditionedResidual()
