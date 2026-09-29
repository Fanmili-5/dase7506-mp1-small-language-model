"""Context-conditioned ByteLevel-token morphology residual for MP1."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TOKENIZER = Path(__file__).resolve().parent / "data/tokenizer.json"
TOKENIZER_SHA256 = "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e"
VOCAB = 2048
MAX_SYMBOLS = 12
SYMBOLS = 256
HIDDEN_WIDTH = 288
RESIDUAL_WIDTH = 128


def fixed_token_symbols() -> tuple[torch.Tensor, torch.Tensor]:
    raw = TOKENIZER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Fixed course tokenizer changed")
    vocab = json.loads(raw)["model"]["vocab"]
    if len(vocab) != VOCAB or set(vocab.values()) != set(range(VOCAB)):
        raise ValueError("Unexpected ByteLevel token IDs")
    alphabet = sorted(set("".join(vocab)))
    if len(alphabet) != SYMBOLS:
        raise ValueError("Unexpected tokenizer symbol alphabet")
    symbol_id = {symbol: index for index, symbol in enumerate(alphabet)}
    ids = torch.zeros(VOCAB, MAX_SYMBOLS, dtype=torch.long)
    mask = torch.zeros(VOCAB, MAX_SYMBOLS, dtype=torch.float32)
    for token, token_id in vocab.items():
        if not 1 <= len(token) <= MAX_SYMBOLS:
            raise ValueError("Unexpected ByteLevel token length")
        for position, symbol in enumerate(token):
            ids[token_id, position] = symbol_id[symbol]
            mask[token_id, position] = 1.0
    return ids, mask


class MorphologyResidual(nn.Module):
    """A zero-start nonlinear residual over the unchanged 2,048 token IDs."""

    def __init__(self):
        super().__init__()
        ids, mask = fixed_token_symbols()
        self.register_buffer("token_symbol_ids", ids, persistent=False)
        self.register_buffer("token_symbol_mask", mask, persistent=False)
        self.register_buffer("token_length_scale", mask.sum(-1).rsqrt(),
                             persistent=False)
        self.symbol_vectors = nn.Parameter(
            torch.empty(MAX_SYMBOLS, SYMBOLS, RESIDUAL_WIDTH))
        nn.init.normal_(self.symbol_vectors, std=0.02)
        self.pre = nn.LayerNorm(HIDDEN_WIDTH)
        self.input = nn.Linear(HIDDEN_WIDTH, RESIDUAL_WIDTH)
        self.output = nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH, bias=False)
        nn.init.zeros_(self.output.weight)

    def token_rows(self) -> torch.Tensor:
        position = torch.arange(MAX_SYMBOLS, device=self.symbol_vectors.device)
        parts = self.symbol_vectors[position[None, :], self.token_symbol_ids]
        rows = (parts * self.token_symbol_mask[..., None]).sum(1)
        return rows * self.token_length_scale[:, None]

    def forward(self, base_logp: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if (base_logp.ndim != 3 or base_logp.shape[-1] != VOCAB
                or hidden.shape != (*base_logp.shape[:2], HIDDEN_WIDTH)
                or base_logp.device != hidden.device):
            raise ValueError("Expected matching causal hidden and 2,048-way base rows")
        context = self.output(F.silu(self.input(self.pre(hidden.float()))))
        delta = context @ self.token_rows().T / math.sqrt(RESIDUAL_WIDTH)
        return F.log_softmax(base_logp.float() + delta.float(), dim=-1)


class FrozenBaseWithMorphology(nn.Module):
    """GPU training/diagnostic wrapper; not the final compact CPU export."""

    def __init__(self, base: nn.Module, residual: MorphologyResidual):
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
        return self.residual(base_logp, hidden)


def build_model(config: dict) -> MorphologyResidual:
    if config != {"width": HIDDEN_WIDTH, "residual_width": RESIDUAL_WIDTH,
                  "tokenizer_sha256": TOKENIZER_SHA256}:
        raise ValueError("Unexpected Stage160 morphology configuration")
    return MorphologyResidual()
