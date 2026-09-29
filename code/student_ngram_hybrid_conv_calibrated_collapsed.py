"""Collapsed MKN mixture with materialized Stage79 scalar calibration."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

import student_ngram_hybrid_conv_bias_collapsed


class CalibratedOutputBiasCollapsedLM(
        student_ngram_hybrid_conv_bias_collapsed.OutputBiasCollapsedLM):
    """Apply vocabulary temperature without untying the input/output matrix.

    The exported output bias already contains `old_bias / temperature` plus the
    selected train-unigram correction, and the copy-gate bias already contains
    its selected shift.  Scaling only the hidden input to the tied vocabulary
    head therefore reproduces the screened logits while leaving copy features
    unchanged.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.vocabulary_temperature = float(config["vocabulary_temperature"])
        if (not math.isfinite(self.vocabulary_temperature)
                or self.vocabulary_temperature <= 0):
            raise ValueError("Vocabulary temperature must be finite and positive")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(
                self.neural.head(hidden / self.vocabulary_temperature)
                + self.neural.output_bias,
                dim=-1,
            )
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden)
            result.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            result.addcmul_(copy, torch.sigmoid(gate) * (1 - self.weight))
            self.ngram.add_into(result, ids, self.weight)
            result.div_(result.sum(dim=-1, keepdim=True))
            return result.log()

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> CalibratedOutputBiasCollapsedLM:
    if config.get("kind") != "hybrid_calibrated_conv":
        raise ValueError("Expected a calibrated hybrid-conv checkpoint")
    return CalibratedOutputBiasCollapsedLM(config)
