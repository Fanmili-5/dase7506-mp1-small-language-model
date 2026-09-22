"""Collapsed modified-KN hybrid with fixed legal scalar calibration."""
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import student_ngram_collapsed

PARENT_SHA = "c3803889f9bc9fcd5c3e1076eef2aa60852380ec9ab91c56f7c0a47e7a9d31a6"


class CalibratedCollapsedHybridLM(student_ngram_collapsed.CollapsedHybridLM):
    def __init__(self, config):
        super().__init__(config)
        self.vocabulary_temperature = float(config["vocabulary_temperature"])
        self.unigram_prior_weight = float(config["unigram_prior_weight"])
        self.copy_gate_shift = float(config["copy_gate_shift"])
        prior = torch.tensor(config["calibration_log_prior"], dtype=torch.float32)
        if (not math.isfinite(self.vocabulary_temperature) or self.vocabulary_temperature <= 0
                or not math.isfinite(self.unigram_prior_weight)
                or not math.isfinite(self.copy_gate_shift)
                or prior.shape != (self.vocab,) or not torch.isfinite(prior).all()):
            raise ValueError("Invalid fixed calibration parameters")
        self.register_buffer("calibration_log_prior", prior)

    def _neural_log_probs(self, hidden, ids):
        vocabulary = F.log_softmax(
            self.neural.head(hidden / self.vocabulary_temperature)
            + self.unigram_prior_weight * self.calibration_log_prior,
            dim=-1,
        )
        copy = self.neural.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        gate = self.neural.copy_gate(hidden) + self.copy_gate_shift
        return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                               F.logsigmoid(gate) + log_copy)

    def forward(self, ids):
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            if self.training:
                neural = self._neural_log_probs(hidden, ids)
                return torch.logaddexp(neural + math.log1p(-self.weight),
                                       self.ngram(ids) + math.log(self.weight))
            result = F.softmax(
                self.neural.head(hidden / self.vocabulary_temperature)
                + self.unigram_prior_weight * self.calibration_log_prior,
                dim=-1,
            )
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden) + self.copy_gate_shift
            result.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            result.addcmul_(copy, torch.sigmoid(gate) * (1 - self.weight))
            self.ngram.add_into(result, ids, self.weight)
            result.div_(result.sum(dim=-1, keepdim=True))
            return result.log()


def build_model(config):
    if hashlib.sha256(Path(student_ngram_collapsed.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Collapsed parent source changed")
    if config.get("kind") != "hybrid_calibrated":
        raise ValueError("This module exports calibrated hybrid inference only")
    return CalibratedCollapsedHybridLM(config)
