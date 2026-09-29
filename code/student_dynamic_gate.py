"""Input-only learned gate between the causal neural and train-count experts."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_ngram_collapsed

PARENT_SHA = "c3803889f9bc9fcd5c3e1076eef2aa60852380ec9ab91c56f7c0a47e7a9d31a6"
MIN_WEIGHT = 1e-4


class DynamicGateLM(student_ngram_collapsed.CollapsedHybridLM):
    def __init__(self, config):
        super().__init__(config)
        width = int(config["neural_config"]["width"])
        self.count_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.count_gate.weight)
        initial = float(config.get("initial_count_weight", self.weight))
        if not MIN_WEIGHT < initial < 1-MIN_WEIGHT:
            raise ValueError("Initial count weight must stay inside the bounded gate")
        probability = (initial-MIN_WEIGHT)/(1-2*MIN_WEIGHT)
        nn.init.constant_(self.count_gate.bias, math.log(probability/(1-probability)))
        self.count_probability_lower_bound *= MIN_WEIGHT / self.weight

    def _loaded(self, module, incompatible_keys):
        super()._loaded(module, incompatible_keys)
        if hasattr(self,"count_gate"):
            if any(not torch.isfinite(p).all() for p in self.count_gate.parameters()):
                raise ValueError("Dynamic gate parameters must be finite")
            self.count_probability_lower_bound *= MIN_WEIGHT / self.weight

    def dynamic_weight(self, hidden):
        return MIN_WEIGHT + (1-2*MIN_WEIGHT)*torch.sigmoid(self.count_gate(hidden))

    def forward(self, ids):
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.softmax(self.neural.head(hidden), dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            copy_gate = self.neural.copy_gate(hidden)
            neural = vocabulary*torch.sigmoid(-copy_gate) + copy*torch.sigmoid(copy_gate)
            count = self.ngram.distribution(ids)
            weight = self.dynamic_weight(hidden)
            result = neural*(1-weight) + count*weight
            result = result/result.sum(-1,keepdim=True)
            return result.log()


def build_model(config):
    if hashlib.sha256(Path(student_ngram_collapsed.__file__).read_bytes()).hexdigest()!=PARENT_SHA:
        raise ValueError("Collapsed hybrid parent changed")
    if config.get("kind")!="hybrid_dynamic_gate":
        raise ValueError("Expected hybrid_dynamic_gate")
    return DynamicGateLM(config)
