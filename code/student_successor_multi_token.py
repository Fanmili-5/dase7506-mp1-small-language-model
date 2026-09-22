"""Stage40 training-only deep supervision and MTP with successor copying."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

import student_multi_token
import student_successor_only

PARENT_SHA256 = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"


class SuccessorMultiTokenLM(student_multi_token.MultiTokenLM):
    copy_distribution = student_successor_only.SuccessorOnlyLM.copy_distribution

    def _prefix_log_probs(self, hidden, ids):
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
            return self._prefix_log_probs(hidden.float(), ids)

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Multi-token parent source changed")
    if config.get("copy_semantics") != "successor":
        raise ValueError("Successor training requires copy_semantics='successor'")
    return SuccessorMultiTokenLM(config)
