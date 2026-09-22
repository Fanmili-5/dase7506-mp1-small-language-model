"""Numerically equivalent calibrated hybrid with temperature folded before projection."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

import student_calibrated_collapsed

PARENT_SHA = "64a20eeffb4457f4a0bd3b1b8d13be7576e877e6b1f8a8898a7cc0a151f7f546"


class FastCalibratedHybridLM(student_calibrated_collapsed.CalibratedCollapsedHybridLM):
    def forward(self, ids):
        if self.training:
            return super().forward(ids)
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            # The vocabulary projection has no bias, so W(h/T) is the same
            # real-valued function as Wh/T while dividing 256 rather than 2048 values.
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
    if hashlib.sha256(Path(student_calibrated_collapsed.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Exact calibrated parent source changed")
    if config.get("kind") != "hybrid_calibrated_fast":
        raise ValueError("This module exports fast calibrated hybrid inference only")
    return FastCalibratedHybridLM(config)
