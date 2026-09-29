"""Single-route causal successor cache for resource-matched inference.

At prediction position t, key h_j retrieves value x_(j+1) only for j < t.
Every value is therefore already present in the current independent input
window.  No cache or state survives a forward call.
"""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

import student_structured

PARENT_SHA256 = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class SuccessorOnlyLM(student_structured.StructuredLM):
    def __init__(self, config):
        if config.get("output_kind") != "prefix_copy":
            raise ValueError("Successor-only model requires the prefix-copy parameterization")
        if config.get("copy_semantics") != "successor":
            raise ValueError("Successor-only model requires copy_semantics='successor'")
        super().__init__(config)

    def copy_distribution(self, hidden, ids):
        query = self.copy_query(hidden).float()
        key = self.copy_key(hidden).float()
        scores = (query @ key.transpose(-1, -2)) * self.copy_scale
        length = ids.shape[1]
        allowed = torch.ones(length, length, device=ids.device, dtype=torch.bool).tril(-1)
        # Keep the empty first row finite, then remove all of its mass.  The
        # mixture gate is disabled at that position in _successor_log_probs.
        allowed[0, 0] = True
        attention = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        attention = attention * (torch.arange(length, device=ids.device) > 0)[None, :, None]
        values = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        probabilities = attention.new_zeros(*ids.shape, self.vocab)
        return probabilities.scatter_add(
            -1, values[:, None, :].expand(-1, length, -1), attention)

    def _successor_log_probs(self, hidden, ids):
        vocabulary = F.log_softmax(self.head(hidden), dim=-1)
        copy = self.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        first = (torch.arange(ids.shape[1], device=ids.device) == 0)[None, :, None]
        gate = self.copy_gate(hidden).masked_fill(first, -float("inf"))
        return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                               F.logsigmoid(gate) + log_copy)

    def forward(self, ids):
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self._successor_log_probs(hidden.float(), ids)

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    actual = hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Structured parent source changed")
    return SuccessorOnlyLM(config)
