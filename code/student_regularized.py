"""Training-only regularizers for the unchanged causal prefix-copy Transformer.

Input embedding row dropout never masks the tied output vocabulary weights.
Hidden dropout acts inside SwiGLU, not on its residual output. Both are removed
by export; this module adds no parameters or persistent evaluation state.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import student
import student_structured

STRUCTURED_SHA = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"
REGULARIZATION_KEYS = ("embedding_row_dropout", "ffn_hidden_dropout")


class HiddenDropSwiGLU(nn.Module):
    def __init__(self, original, probability):
        super().__init__()
        # Reuse the initialized parameters and state keys, without consuming RNG.
        self.input, self.output = original.input, original.output
        self.probability = probability

    def forward(self, x):
        gate, value = self.input(x).chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        return self.output(F.dropout(hidden, self.probability, self.training))


class RegularizedLM(student_structured.StructuredLM):
    def __init__(self, config):
        probabilities = [float(config.get(key, 0)) for key in REGULARIZATION_KEYS]
        if any(not math.isfinite(p) or not 0 <= p < 1 for p in probabilities):
            raise ValueError("Regularizer probabilities must be finite and in [0,1)")
        super().__init__(config)
        self.embedding_row_dropout, self.ffn_hidden_dropout = probabilities
        if self.ffn_hidden_dropout:
            for block in self.blocks:
                if not isinstance(block.mlp, student.SwiGLU):
                    raise ValueError("Hidden dropout requires SwiGLU")
                block.mlp = HiddenDropSwiGLU(block.mlp, self.ffn_hidden_dropout)

    def input_embeddings(self, ids):
        if not self.training or self.embedding_row_dropout == 0:
            return self.token(ids)
        # One independent row mask per call. Its shape does not depend on token
        # content, so changing a future token cannot change a prefix's mask.
        mask = F.dropout(self.token.weight.new_ones(self.vocab, 1),
                         self.embedding_row_dropout, training=True)
        return F.embedding(ids, self.token.weight * mask)

    def features(self, ids):
        if not self.training or self.embedding_row_dropout == 0:
            return super().features(ids)
        if ids.ndim != 2 or ids.shape[1] > self.context:
            raise ValueError("ids must be [batch,time<=256]")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def inference_config(config):
    return {key: value for key, value in config.items() if key not in REGULARIZATION_KEYS}


def build_model(config):
    if hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest() != STRUCTURED_SHA:
        raise ValueError("Structured model source hash mismatch")
    if hashlib.sha256(Path(student.__file__).read_bytes()).hexdigest() != student_structured.BACKBONE_SHA256:
        raise ValueError("Backbone source hash mismatch")
    return RegularizedLM(config)
