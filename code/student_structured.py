"""Additive MP1 prefix-copy and mixture-of-softmax heads.

The unchanged backbone is hash-pinned so its dependency cannot silently change
under the official evaluator's implementation-module hash. Include both source
files in every checkpoint bundle. No external or cross-window state is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student

BACKBONE_SHA256 = "e5dc885178e1ebcc4b89215ca719ecd83687709db879cd5eafc4e504b0c67dec"


class StructuredLM(student.StudentLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.output_kind = str(config["output_kind"])
        width = int(config["width"])
        if self.output_kind == "prefix_copy":
            size = int(config.get("copy_dim", 64))
            if size <= 0:
                raise ValueError("copy_dim must be positive")
            self.copy_query = nn.Linear(width, size, bias=False)
            self.copy_key = nn.Linear(width, size, bias=False)
            self.copy_gate = nn.Linear(width, 1)
            self.copy_query.apply(self._initialize)
            self.copy_key.apply(self._initialize)
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, float(config.get("copy_gate_bias", -2.0)))
            self.copy_scale = size ** -0.5
        elif self.output_kind == "mos":
            self.components = int(config.get("mos_components", 2))
            if self.components < 1:
                raise ValueError("mos_components must be positive")
            self.mos_projection = nn.Linear(width, self.components * width)
            self.mos_gate = nn.Linear(width, self.components)
            self.mos_projection.apply(self._initialize)
            nn.init.zeros_(self.mos_gate.weight)
            nn.init.zeros_(self.mos_gate.bias)
        else:
            raise ValueError(f"Unknown output_kind: {self.output_kind}")

    def copy_distribution(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Position t may copy only x_0,...,x_t, never targets or future ids."""
        query = self.copy_query(hidden).float()
        key = self.copy_key(hidden).float()
        scores = (query @ key.transpose(-1, -2)) * self.copy_scale
        length = ids.shape[1]
        future = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
        attention = scores.masked_fill(future, float("-inf")).softmax(-1)
        probabilities = attention.new_zeros(*ids.shape, self.vocab)
        return probabilities.scatter_add(-1, ids[:, None, :].expand(-1, length, -1), attention)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        # Heads stay FP32 even during BF16 backbone training. This avoids underflow
        # in the mixture and makes cross_entropy(log p, y) equivalent to NLL.
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            if self.output_kind == "prefix_copy":
                vocabulary = F.log_softmax(self.head(hidden), dim=-1)
                copy = self.copy_distribution(hidden, ids)
                # Avoid log(0) gradients. Missing tokens still have exactly zero
                # copy probability; the vocabulary branch makes the result finite.
                log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
                log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
                gate = self.copy_gate(hidden)
                return torch.logaddexp(
                    F.logsigmoid(-gate) + vocabulary,
                    F.logsigmoid(gate) + log_copy,
                )
            shape = (*hidden.shape[:2], self.components, hidden.shape[-1])
            projected = self.mos_projection(hidden).view(shape).tanh()
            component_logp = F.log_softmax(self.head(projected), dim=-1)
            log_weights = F.log_softmax(self.mos_gate(hidden), dim=-1)
            return torch.logsumexp(component_logp + log_weights.unsqueeze(-1), dim=-2)

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> StructuredLM:
    actual = hashlib.sha256(Path(student.__file__).read_bytes()).hexdigest()
    if actual != BACKBONE_SHA256:
        raise ValueError("Backbone source hash mismatch; restore the pinned student.py")
    return StructuredLM(config)
