"""Collapsed modified-Kneser-Ney mixture for the Stage46 MoE neural expert."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import student_moe_structured
from student_ngram_collapsed import CollapsedNgramLM


class MoECollapsedHybridLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.neural = student_moe_structured.build_model(config["neural_config"])
        self.ngram = CollapsedNgramLM(config)
        self.context, self.vocab = 256, 2048
        self.weight = float(config["mixture_weight"])
        if not 0 < self.weight < 1:
            raise ValueError("Collapsed hybrid requires both experts")
        self.register_load_state_dict_post_hook(self._loaded)

    def _loaded(self, module, incompatible_keys):
        unigram = self.ngram.unigram
        if not torch.isfinite(unigram).all() or unigram.min() <= 0:
            raise ValueError("Unigram must have finite positive support")
        bound = self.weight * float(unigram.min())
        for table in self.ngram.tables:
            if not table.backoff.numel():
                continue
            if (not torch.isfinite(table.backoff).all() or table.backoff.min() <= 0
                    or table.backoff.max() > 1 or not torch.isfinite(table.mass).all()
                    or (table.mass < 0).any()):
                raise ValueError("Invalid positive-backoff statistics")
            bound *= float(table.backoff.min())
        if bound <= 16 * torch.finfo(torch.float32).tiny:
            raise ValueError("Count floor too small for safe FP32 probability mixture")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(self.neural.head(hidden), dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden)
            result.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            result.addcmul_(copy, torch.sigmoid(gate) * (1 - self.weight))
            self.ngram.add_into(result, ids, self.weight)
            result.div_(result.sum(dim=-1, keepdim=True))
            return result.log()

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> MoECollapsedHybridLM:
    if config.get("kind") != "hybrid":
        raise ValueError("This module exports hybrid inference only")
    return MoECollapsedHybridLM(config)
