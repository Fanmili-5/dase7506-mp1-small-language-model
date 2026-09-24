"""Hybrid-conv neural expert with a single causal successor-copy route."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_hybrid_conv_output_bias
from student_stage135_successor_gate import add_successor_copy


class SuccessorHybridConvLM(student_hybrid_conv_output_bias.HybridConvOutputBiasLM):
    def copy_distribution(self, hidden: torch.Tensor,
                          ids: torch.Tensor) -> torch.Tensor:
        query = self.copy_query(hidden).float()
        key = self.copy_key(hidden).float()
        scores = (query @ key.transpose(-1, -2)) * self.copy_scale
        batch, length = ids.shape
        copy = scores.new_zeros(batch, length, self.vocab)
        scale = scores.new_ones(batch, length, 1)
        return add_successor_copy(copy, ids, scores, scale)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.softmax(self.head(hidden) + self.output_bias, dim=-1)
            copy = self.copy_distribution(hidden, ids)
            gate = torch.sigmoid(self.copy_gate(hidden))
            first = torch.arange(ids.shape[1], device=ids.device).eq(0)[None, :, None]
            gate = torch.where(first, 0., gate)
            return ((1 - gate) * vocabulary + gate * copy).log()


def calibrated_successor_log_probs(neural: SuccessorHybridConvLM,
                                   ids: torch.Tensor, log_prior: torch.Tensor,
                                   temperature: float = 1.125,
                                   prior_weight: float = .0625,
                                   copy_shift: float = .25,
                                   return_gate: bool = False):
    hidden = neural.features(ids)
    with torch.autocast(device_type=ids.device.type, enabled=False):
        hidden = hidden.float()
        vocabulary = F.softmax(
            (neural.head(hidden) + neural.output_bias) / temperature
            + prior_weight * log_prior, dim=-1)
        copy = neural.copy_distribution(hidden, ids)
        gate = torch.sigmoid(neural.copy_gate(hidden) + copy_shift)
        first = torch.arange(ids.shape[1], device=ids.device).eq(0)[None, :, None]
        gate = torch.where(first, 0., gate)
        log_probs = ((1 - gate) * vocabulary + gate * copy).log()
        return (log_probs, gate) if return_gate else log_probs


def build_model(config: dict) -> SuccessorHybridConvLM:
    return SuccessorHybridConvLM(config)
