"""R-Drop training view of the fixed multi-token prefix-copy Transformer."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import student_multi_token

PARENT_SHA = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"
RDROP_KEYS = ("rdrop_alpha",)


class RDropMultiTokenLM(student_multi_token.MultiTokenLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.rdrop_alpha = float(config["rdrop_alpha"])
        if not math.isfinite(self.rdrop_alpha) or self.rdrop_alpha <= 0:
            raise ValueError("rdrop_alpha must be finite and positive")

    def single_training_pass(self, ids, targets, future_targets):
        """Return the ordinary objective and its primary normalized log-probs."""
        if not self.training:
            raise ValueError("single_training_pass requires train mode")
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
                for index, (norm, projection) in enumerate(
                    zip(self.future_norms, self.future_projections))
            ]).mean()
            total = primary + self.deep_supervision_weight * deep + self.future_prediction_weight * future
        return total, primary_logp, {
            "primary": primary.detach(), "deep": deep.detach(), "future": future.detach()
        }

    def rdrop_training_loss(self, ids, targets, future_targets):
        first_loss, first_logp, first_parts = self.single_training_pass(
            ids, targets, future_targets)
        second_loss, second_logp, second_parts = self.single_training_pass(
            ids, targets, future_targets)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            first_probability = first_logp.exp()
            second_probability = second_logp.exp()
            symmetric_kl = 0.5 * (
                (first_probability * (first_logp - second_logp)).sum(dim=-1).mean()
                + (second_probability * (second_logp - first_logp)).sum(dim=-1).mean()
            )
            total = 0.5 * (first_loss + second_loss) + self.rdrop_alpha * symmetric_kl
        parts = {
            key: 0.5 * (first_parts[key] + second_parts[key])
            for key in first_parts
        }
        parts["symmetric_kl"] = symmetric_kl.detach()
        return total, parts


def inference_config(config: dict) -> dict:
    result = student_multi_token.inference_config(config)
    for key in RDROP_KEYS:
        result.pop(key, None)
    return result


def inference_state(state: dict) -> dict:
    return student_multi_token.inference_state(state)


def build_model(config: dict) -> RDropMultiTokenLM:
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Multi-token parent source changed")
    return RDropMultiTokenLM(config)
