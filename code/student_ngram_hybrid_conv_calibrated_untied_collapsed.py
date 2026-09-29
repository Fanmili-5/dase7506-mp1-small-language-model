"""Collapsed calibrated hybrid with a materialized untied output matrix."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import student_ngram_hybrid_conv_bias_collapsed


class UntiedCalibratedOutputBiasCollapsedLM(
        student_ngram_hybrid_conv_bias_collapsed.OutputBiasCollapsedLM):
    """Use a separately scaled vocabulary head while preserving input embeddings."""

    def __init__(self, config: dict):
        super().__init__(config)
        width = int(config["neural_config"]["width"])
        self.neural.head = nn.Linear(width, self.vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(
                self.neural.head(hidden) + self.neural.output_bias, dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden)
            result.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            result.addcmul_(copy, torch.sigmoid(gate) * (1 - self.weight))
            self.ngram.add_into(result, ids, self.weight)
            result.div_(result.sum(dim=-1, keepdim=True))
            return result.log()

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> UntiedCalibratedOutputBiasCollapsedLM:
    if config.get("kind") != "hybrid_calibrated_untied_conv":
        raise ValueError("Expected an untied calibrated hybrid-conv checkpoint")
    return UntiedCalibratedOutputBiasCollapsedLM(config)
