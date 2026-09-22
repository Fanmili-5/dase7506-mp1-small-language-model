"""Training-only multi-token prediction on top of Stage22 deep supervision.

Every auxiliary head reads the same causal hidden state and predicts a later
token from supplied training text. Auxiliary modules are removed at export.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_deep_supervision

PARENT_SHA = "9a80bc4463059dc35c06825a761ac9ea5a5c538bffcedc59e6055dafe79c3ee0"
FUTURE_KEYS = ("future_prediction_offsets", "future_prediction_weight")


class IdentityProjection(nn.Module):
    """Trainable identity initialization without consuming random numbers."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))

    def forward(self, inputs):
        return F.linear(inputs, self.weight)


class MultiTokenLM(student_deep_supervision.DeepSupervisionLM):
    def __init__(self, config):
        super().__init__(config)
        offsets = tuple(int(value) for value in config.get("future_prediction_offsets", ()))
        weight = float(config.get("future_prediction_weight", 0))
        if not offsets or tuple(sorted(set(offsets))) != offsets or offsets[0] < 2:
            raise ValueError("Future offsets must be unique, increasing, and start at 2 or later")
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("Future prediction weight must lie in (0,1]")
        self.future_prediction_offsets = offsets
        self.future_prediction_weight = weight
        width = int(config["width"])
        eps = float(config.get("norm_eps", 1e-5))
        self.future_norms = nn.ModuleList([
            student.make_norm(str(config.get("norm", "layernorm")).lower(), width, eps)
            for _ in offsets
        ])
        self.future_projections = nn.ModuleList([IdentityProjection(width) for _ in offsets])

    def training_loss(self, ids, targets, future_targets):
        if not self.training:
            raise ValueError("training_loss requires train mode")
        if ids.shape != targets.shape or ids.ndim != 2:
            raise ValueError("Inputs and primary targets must share [batch,time]")
        expected = (len(self.future_prediction_offsets), *ids.shape)
        if tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        auxiliary_logits = []
        auxiliary_index = 0
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if (auxiliary_index < len(self.deep_supervision_layers)
                    and layer == self.deep_supervision_layers[auxiliary_index]):
                auxiliary_logits.append(self.head(self.auxiliary_norms[auxiliary_index](x)))
                auxiliary_index += 1
        hidden = self.norm(x)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            primary_logp = self._prefix_log_probs(hidden.float(), ids)
            primary = F.nll_loss(primary_logp.flatten(0, 1), targets.flatten())
            deep = torch.stack([
                F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
                for logits in auxiliary_logits
            ]).mean()
            future = torch.stack([
                F.cross_entropy(
                    self.head(projection(norm(hidden.float()))).flatten(0, 1),
                    future_targets[index].flatten(),
                )
                for index, (norm, projection) in enumerate(zip(self.future_norms, self.future_projections))
            ]).mean()
            total = primary + self.deep_supervision_weight * deep + self.future_prediction_weight * future
        return total, {"primary": primary.detach(), "deep": deep.detach(), "future": future.detach()}


def inference_config(config):
    return {key: value for key, value in student_deep_supervision.inference_config(config).items()
            if key not in FUTURE_KEYS}


def inference_state(state):
    stripped = student_deep_supervision.inference_state(state)
    return {key: value for key, value in stripped.items()
            if not key.startswith(("future_norms.", "future_projections."))}


def build_model(config):
    if hashlib.sha256(Path(student_deep_supervision.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Deep-supervision parent source changed")
    return MultiTokenLM(config)
