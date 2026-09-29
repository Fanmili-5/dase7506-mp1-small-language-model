"""Resource-aware probability ensemble for the fixed MP1 evaluator.

The ensemble combines normalized member distributions rather than logits.  Its
members use the existing ``student.py`` implementation, so the predictor stays
within the same causal, stateless 256-token contract.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from student import build_model as build_student


class StudentEnsemble(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.context = int(config["context"])
        self.vocab = int(config["vocab"])
        member_configs = list(config.get("member_configs", []))
        weights = [float(value) for value in config.get("weights", [])]
        if len(member_configs) != 2 or len(weights) != 2:
            raise ValueError("The resource-screened ensemble requires exactly two members.")
        if any(value <= 0.0 for value in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Ensemble weights must be positive and sum to one.")
        for member in member_configs:
            if int(member["context"]) != self.context or int(member["vocab"]) != self.vocab:
                raise ValueError("All members must share the ensemble context and vocabulary.")
        self.members = nn.ModuleList(build_student(member) for member in member_configs)
        self.register_buffer("log_weights", torch.tensor(weights, dtype=torch.float64).log())

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        combined = self.members[0].predict_log_probs(ids) + self.log_weights[0].to(ids.device)
        other = self.members[1].predict_log_probs(ids) + self.log_weights[1].to(ids.device)
        return torch.logaddexp(combined, other)


def build_model(config: dict) -> StudentEnsemble:
    return StudentEnsemble(config)
