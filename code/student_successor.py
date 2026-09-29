"""Validation candidates: precision-matched control and two causal copy routes.

Successor memory uses key h_j and value x_(j+1), with j < t when predicting
x_(t+1). Every accessible value is therefore already an input. No state persists
between calls. This is an independent-window adaptation, not an external cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_structured

STRUCTURED_SHA256 = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class SuccessorLM(student_structured.StructuredLM):
    def __init__(self, config):
        if config.get("output_kind") != "prefix_copy":
            raise ValueError("Successor candidates require prefix_copy base configuration")
        super().__init__(config)
        self.experiment_kind = config["experiment_kind"]
        if self.experiment_kind == "fp32_control":
            # Consume precisely the same initialization RNG as B, then remove
            # the branch. This keeps the backbone initialization and subsequent
            # dropout RNG aligned with B, without inactive trainable parameters.
            del self.copy_query, self.copy_key, self.copy_gate
        elif self.experiment_kind == "dual_copy":
            width, size = int(config["width"]), int(config.get("copy_dim", 64))
            self.successor_query = nn.Linear(width, size, bias=False)
            self.successor_key = nn.Linear(width, size, bias=False)
            self.successor_gate = nn.Linear(width, 1)
            self.successor_query.apply(self._initialize)
            self.successor_key.apply(self._initialize)
            nn.init.zeros_(self.successor_gate.weight)
            nn.init.constant_(self.successor_gate.bias, -2.0)
        else:
            raise ValueError(f"Unknown experiment_kind: {self.experiment_kind}")

    def successor_distribution(self, hidden, ids):
        length = ids.shape[1]
        query, key = self.successor_query(hidden), self.successor_key(hidden)
        scores = (query @ key.transpose(-1, -2)) * self.copy_scale
        allowed = torch.ones(length, length, device=ids.device, dtype=torch.bool).tril(-1)
        # Empty row 0 needs a finite softmax for stable backward. Its probability
        # mass is zeroed and its mixture gate disabled below, even for length=1.
        allowed[0, 0] = True
        attention = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        usable = (torch.arange(length, device=ids.device) > 0)[None, :, None]
        attention = attention * usable
        values = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        probabilities = attention.new_zeros(*ids.shape, self.vocab)
        return probabilities.scatter_add(-1, values[:, None, :].expand(-1, length, -1), attention)

    @staticmethod
    def probability_log(probabilities):
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log().masked_fill(
            probabilities == 0, float("-inf"))

    def forward(self, ids):
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.log_softmax(self.head(hidden), dim=-1)
            if self.experiment_kind == "fp32_control":
                return vocabulary
            content = self.probability_log(self.copy_distribution(hidden, ids))
            successor = self.probability_log(self.successor_distribution(hidden, ids))
            raw_successor_gate = self.successor_gate(hidden)
            first = (torch.arange(ids.shape[1], device=ids.device) == 0)[None, :, None]
            successor_gate = raw_successor_gate.masked_fill(first, float("-inf"))
            gates = F.log_softmax(torch.cat((torch.zeros_like(successor_gate),
                                             self.copy_gate(hidden), successor_gate), dim=-1), -1)
            # logsumexp over all routes avoids the undefined (-inf,-inf)
            # logaddexp gradient for vocabulary entries absent from both routes.
            routes = torch.stack((vocabulary, content, successor), dim=-1)
            return torch.logsumexp(routes + gates.unsqueeze(-2), dim=-1)


def build_model(config):
    if hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest() != STRUCTURED_SHA256:
        raise ValueError("Structured source hash mismatch")
    if hashlib.sha256(Path(student.__file__).read_bytes()).hexdigest() != student_structured.BACKBONE_SHA256:
        raise ValueError("Backbone source hash mismatch")
    return SuccessorLM(config)
