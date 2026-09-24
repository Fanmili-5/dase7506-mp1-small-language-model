"""Causal calibrated neural/order-6 MKN mixture with a four-feature gate."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_output_bias
from student_ngram_collapsed import CollapsedNgramLM


class GatedHybridLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if (config.get("kind") != "gated_hybrid"
                or config.get("vocab") != 2048
                or config.get("context") != 256
                or config.get("max_order") != 6
                or config.get("feature_names") != [
                    "neural_max_logp", "neural_margin", "highest_backoff",
                    "highest_max_mass"]):
            raise ValueError("Unexpected gated-hybrid configuration")
        self.neural = student_hybrid_conv_output_bias.build_model(
            config["neural_config"])
        self.ngram = CollapsedNgramLM(config)
        self.context, self.vocab = 256, 2048
        self.temperature = float(config["vocabulary_temperature"])
        self.prior_weight = float(config["train_unigram_prior_weight"])
        self.copy_shift = float(config["copy_gate_shift"])
        self.slope_scale = float(config["slope_scale"])
        anchor = float(config["anchor_weight"])
        if not (0 < anchor < 1 and self.temperature > 0
                and self.slope_scale >= 0):
            raise ValueError("Invalid gate scalars")
        self.anchor_logit = math.log(anchor / (1 - anchor))
        self.register_buffer("log_prior", torch.zeros(self.vocab))
        self.register_buffer("gate_mean", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("gate_std", torch.ones(4, dtype=torch.float64))
        self.register_buffer("gate_coeff", torch.zeros(4, dtype=torch.float64))

    def _sparse_confidence(self, ids):
        """The two highest-order sparse features used by Stage102."""
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device).expand(batch, -1).flatten()
        backoff = torch.ones(ids.numel(), device=ids.device)
        max_mass = torch.zeros(ids.numel(), device=ids.device)
        for history, table in enumerate(self.ngram.tables, 1):
            if history > length or table.keys.numel() == 0:
                continue
            keys = torch.zeros_like(ids)
            for lag in range(history - 1, -1, -1):
                shifted = ids if lag == 0 else F.pad(ids[:, :-lag], (lag, 0))
                keys = keys * self.vocab + shifted
            keys = keys.flatten()
            locations = torch.searchsorted(table.keys, keys).clamp_max(
                table.keys.numel() - 1)
            found = (table.keys[locations] == keys) & (positions >= history - 1)
            if not found.any():
                continue
            found_rows = found.nonzero().flatten()
            found_locations = locations[found_rows]
            sizes = table.offsets[found_locations + 1] - table.offsets[found_locations]
            starts = table.offsets[found_locations]
            repeated_rows = torch.repeat_interleave(found_rows, sizes)
            repeated_starts = torch.repeat_interleave(starts, sizes)
            local = torch.arange(repeated_rows.numel(), device=ids.device)
            local -= torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            mass = table.mass[repeated_starts + local]
            maxima = torch.zeros(ids.numel(), device=ids.device)
            maxima.scatter_reduce_(0, repeated_rows, mass, reduce="amax",
                                   include_self=True)
            backoff[found_rows] = table.backoff[found_locations]
            max_mass[found_rows] = maxima[found_rows]
        return backoff.reshape(batch, length), max_mass.reshape(batch, length)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent causal windows")
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.log_softmax(
                (self.neural.head(hidden) + self.neural.output_bias)
                / self.temperature + self.prior_weight * self.log_prior,
                dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
            log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
            gate = self.neural.copy_gate(hidden) + self.copy_shift
            neural_logp = torch.logaddexp(
                F.logsigmoid(-gate) + vocabulary,
                F.logsigmoid(gate) + log_copy)
            count_logp = self.ngram.predict_log_probs(ids)
            top = neural_logp.topk(2, dim=-1).values
            backoff, max_mass = self._sparse_confidence(ids)
            features = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                                    backoff, max_mass), dim=-1).double()
            slope = ((features - self.gate_mean) / self.gate_std)
            slope = (slope * self.gate_coeff).sum(-1)
            weight = torch.sigmoid(self.anchor_logit + self.slope_scale * slope)
            weight = 1e-4 + (1 - 2e-4) * weight
            weight = weight.float().unsqueeze(-1)
            return torch.logaddexp(neural_logp + torch.log1p(-weight),
                                   count_logp + weight.log())

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> GatedHybridLM:
    return GatedHybridLM(config)
