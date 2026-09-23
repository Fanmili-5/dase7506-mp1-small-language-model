"""Hybrid-conv inference model with a train-fitted vocabulary intercept."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_structured

PARENT_SHA = "81e61cc36bc6aec3711de27c9860c7ac6b80238b8b5481c5bfa7710a04d9c34e"


class HybridConvOutputBiasLM(student_hybrid_conv_structured.HybridConvStructuredLM):
    """Add one vocabulary bias after the tied output projection."""

    def __init__(self, config: dict):
        if config.get("output_bias") is not True:
            raise ValueError("Expected output_bias=true")
        super().__init__(config)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.log_softmax(self.head(hidden) + self.output_bias, dim=-1)
            copy = self.copy_distribution(hidden, ids)
            log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
            log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
            gate = self.copy_gate(hidden)
            return torch.logaddexp(
                F.logsigmoid(-gate) + vocabulary,
                F.logsigmoid(gate) + log_copy,
            )

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> HybridConvOutputBiasLM:
    actual = hashlib.sha256(
        Path(student_hybrid_conv_structured.__file__).read_bytes()
    ).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Hybrid-conv parent source changed")
    return HybridConvOutputBiasLM(config)
