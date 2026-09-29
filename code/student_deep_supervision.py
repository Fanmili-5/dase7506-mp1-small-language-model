"""Training-only intermediate next-token supervision for the fixed Transformer.

Auxiliary heads see the same causal hidden states and same next-token labels as
the final head. They are removed at export; evaluation is the original model.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_regularized

REGULARIZED_SHA = "36502c9ea3c9cce97f7407889d729af747731169eeede7f587ff3b2b7f84ba13"
AUXILIARY_KEYS = ("deep_supervision_layers", "deep_supervision_weight")


class DeepSupervisionLM(student_regularized.RegularizedLM):
    def __init__(self, config):
        super().__init__(config)
        layers = tuple(int(v) for v in config.get("deep_supervision_layers", ()))
        weight = float(config.get("deep_supervision_weight", 0))
        if not layers or tuple(sorted(set(layers))) != layers:
            raise ValueError("Auxiliary layers must be unique and increasing")
        if layers[0] < 1 or layers[-1] >= len(self.blocks):
            raise ValueError("Auxiliary layers must precede the final block")
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("Deep-supervision weight must lie in (0,1]")
        self.deep_supervision_layers = layers
        self.deep_supervision_weight = weight
        width = int(config["width"])
        eps = float(config.get("norm_eps", 1e-5))
        # Separate training-only norms avoid forcing the deployed final norm to
        # serve incompatible intermediate feature distributions.
        self.auxiliary_norms = nn.ModuleList(
            [student.make_norm(str(config.get("norm", "layernorm")).lower(), width, eps)
             for _ in layers]
        )

    def _prefix_log_probs(self, hidden, ids):
        vocabulary = F.log_softmax(self.head(hidden), dim=-1)
        copy = self.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        gate = self.copy_gate(hidden)
        return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                               F.logsigmoid(gate) + log_copy)

    def training_loss(self, ids, targets):
        if not self.training:
            raise ValueError("training_loss requires train mode")
        if ids.shape != targets.shape or ids.ndim != 2:
            raise ValueError("Inputs and targets must have the same [batch,time] shape")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        auxiliary_logits = []
        auxiliary_index = 0
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if auxiliary_index < len(self.deep_supervision_layers) and layer == self.deep_supervision_layers[auxiliary_index]:
                normalized = self.auxiliary_norms[auxiliary_index](x)
                auxiliary_logits.append(self.head(normalized))
                auxiliary_index += 1
        hidden = self.norm(x)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            primary_logp = self._prefix_log_probs(hidden.float(), ids)
            primary = F.nll_loss(primary_logp.flatten(0, 1), targets.flatten())
            auxiliary = torch.stack([
                F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
                for logits in auxiliary_logits
            ]).mean()
            total = primary + self.deep_supervision_weight * auxiliary
        return total, {"primary": primary.detach(), "auxiliary": auxiliary.detach()}


def inference_config(config):
    return {key: value for key, value in student_regularized.inference_config(config).items()
            if key not in AUXILIARY_KEYS}


def inference_state(state):
    return {key: value for key, value in state.items() if not key.startswith("auxiliary_norms.")}


def build_model(config):
    if hashlib.sha256(Path(student_regularized.__file__).read_bytes()).hexdigest() != REGULARIZED_SHA:
        raise ValueError("Regularized parent source changed")
    return DeepSupervisionLM(config)
