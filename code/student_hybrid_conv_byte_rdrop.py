"""Training-only byte composition for the hybrid-conv R-Drop language model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_byte_composed
import student_hybrid_conv_rdrop

HYBRID_PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
BYTE_PARENT_SHA = "59ae25c00df3d2b46fa20b20feff2e31af267fc289cb1dd1ce84b65bcaf9010b"
BYTE_FEATURE_KEY = "byte_features"


class HybridConvByteRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    """R-Drop training model whose tied vocabulary weights share byte features."""

    def __init__(self, config: dict):
        if config.get(BYTE_FEATURE_KEY) != "bag_first_last":
            raise ValueError("Expected byte_features='bag_first_last'")
        super().__init__(config)
        self.byte_projection = nn.Parameter(torch.zeros(256 * 3, int(config["width"])))
        self.register_buffer(
            "byte_feature_matrix",
            student_byte_composed.build_byte_features(self.vocab),
            persistent=False,
        )

    def composed_weight(self):
        return self.token.weight + self.byte_feature_matrix @ self.byte_projection

    def _input_embeddings_from_weight(self, ids, weight):
        if self.training and self.embedding_row_dropout:
            mask = F.dropout(
                weight.new_ones(self.vocab, 1),
                self.embedding_row_dropout,
                training=True,
            )
            weight = weight * mask
        return F.embedding(ids, weight)

    def input_embeddings(self, ids):
        return self._input_embeddings_from_weight(ids, self.composed_weight())

    def _features_with_weight(self, ids, weight):
        if ids.ndim != 2 or ids.shape[1] > self.context:
            raise ValueError("ids must be [batch,time<=256]")
        x = self._input_embeddings_from_weight(ids, weight)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def features(self, ids):
        return self._features_with_weight(ids, self.composed_weight())

    def _prefix_log_probs(self, hidden, ids, weight=None):
        weight = self.composed_weight() if weight is None else weight
        vocabulary = F.log_softmax(F.linear(hidden, weight), dim=-1)
        copy = self.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        gate = self.copy_gate(hidden)
        return torch.logaddexp(
            F.logsigmoid(-gate) + vocabulary,
            F.logsigmoid(gate) + log_copy,
        )

    def single_training_pass(self, ids, targets, future_targets):
        if not self.training:
            raise ValueError("single_training_pass requires train mode")
        if ids.shape != targets.shape or ids.ndim != 2:
            raise ValueError("Inputs and primary targets must share [batch,time]")
        expected = (len(self.future_prediction_offsets), *ids.shape)
        if tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}")
        weight = self.composed_weight()
        x = self._input_embeddings_from_weight(ids, weight)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        auxiliary_hidden = []
        auxiliary_index = 0
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if (auxiliary_index < len(self.deep_supervision_layers)
                    and layer == self.deep_supervision_layers[auxiliary_index]):
                auxiliary_hidden.append(self.auxiliary_norms[auxiliary_index](x))
                auxiliary_index += 1
        hidden = self.norm(x)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            weight = weight.float()
            primary_logp = self._prefix_log_probs(hidden.float(), ids, weight)
            primary = F.nll_loss(primary_logp.flatten(0, 1), targets.flatten())
            deep = torch.stack([
                F.cross_entropy(
                    F.linear(value.float(), weight).flatten(0, 1),
                    targets.flatten(),
                )
                for value in auxiliary_hidden
            ]).mean()
            future = torch.stack([
                F.cross_entropy(
                    F.linear(projection(norm(hidden.float())), weight).flatten(0, 1),
                    future_targets[index].flatten(),
                )
                for index, (norm, projection) in enumerate(
                    zip(self.future_norms, self.future_projections)
                )
            ]).mean()
            total = (
                primary
                + self.deep_supervision_weight * deep
                + self.future_prediction_weight * future
            )
        return total, primary_logp, {
            "primary": primary.detach(),
            "deep": deep.detach(),
            "future": future.detach(),
        }

    def forward(self, ids):
        weight = self.composed_weight()
        hidden = self._features_with_weight(ids, weight)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self._prefix_log_probs(hidden.float(), ids, weight.float())

    def predict_log_probs(self, ids):
        return self(ids)


def inference_config(config: dict) -> dict:
    result = student_hybrid_conv_rdrop.inference_config(config)
    result.pop(BYTE_FEATURE_KEY, None)
    return result


def inference_state(state: dict) -> dict:
    result = student_hybrid_conv_rdrop.inference_state(state)
    result.pop("byte_projection", None)
    return result


def build_model(config: dict) -> HybridConvByteRDropLM:
    hybrid_sha = hashlib.sha256(
        Path(student_hybrid_conv_rdrop.__file__).read_bytes()
    ).hexdigest()
    byte_sha = hashlib.sha256(Path(student_byte_composed.__file__).read_bytes()).hexdigest()
    if hybrid_sha != HYBRID_PARENT_SHA:
        raise ValueError("Hybrid-conv R-Drop parent source changed")
    if byte_sha != BYTE_PARENT_SHA:
        raise ValueError("Byte-composition source changed")
    return HybridConvByteRDropLM(config)
