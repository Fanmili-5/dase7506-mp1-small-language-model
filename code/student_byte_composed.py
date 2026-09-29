"""Training-only byte-composed BPE embeddings for rare-token sharing."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_multi_token

PARENT_SHA256 = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"
BYTE_FEATURE_KEY = "byte_features"


def byte_unicode_reverse():
    """Inverse of the byte-level BPE reversible alphabet."""
    byte_values = (list(range(ord("!"), ord("~") + 1))
                   + list(range(ord("¡"), ord("¬") + 1))
                   + list(range(ord("®"), ord("ÿ") + 1)))
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


def build_byte_features(vocab):
    tokenizer_path = Path(__file__).with_name("data") / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocabulary = tokenizer["model"]["vocab"]
    if len(vocabulary) != vocab or sorted(vocabulary.values()) != list(range(vocab)):
        raise ValueError("Tokenizer vocabulary does not match the model")
    reverse = byte_unicode_reverse()
    features = torch.zeros(vocab, 256 * 3)
    for token, index in vocabulary.items():
        values = [reverse[character] for character in token]
        if not values:
            raise ValueError("Empty BPE token")
        scale = len(values) ** -0.5
        for value in values:
            features[index, value] += scale
        features[index, 256 + values[0]] = 1.
        features[index, 512 + values[-1]] = 1.
    return features


class ByteComposedLM(student_multi_token.MultiTokenLM):
    def __init__(self, config):
        if config.get(BYTE_FEATURE_KEY) != "bag_first_last":
            raise ValueError("Expected byte_features='bag_first_last'")
        super().__init__(config)
        width = int(config["width"])
        self.byte_projection = nn.Parameter(torch.zeros(256 * 3, width))
        self.register_buffer("byte_feature_matrix", build_byte_features(self.vocab), persistent=False)

    def composed_weight(self):
        return self.token.weight + self.byte_feature_matrix @ self.byte_projection

    def input_embeddings(self, ids):
        table = self.composed_weight()
        return self._input_embeddings_from_weight(ids, table)

    def _input_embeddings_from_weight(self, ids, table):
        if self.training and self.embedding_row_dropout:
            mask = F.dropout(table.new_ones(self.vocab, 1), self.embedding_row_dropout, training=True)
            table = table * mask
        return F.embedding(ids, table)

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
        return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                               F.logsigmoid(gate) + log_copy)

    def training_loss(self, ids, targets, future_targets):
        if not self.training:
            raise ValueError("training_loss requires train mode")
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
                F.cross_entropy(F.linear(value.float(), weight).flatten(0, 1), targets.flatten())
                for value in auxiliary_hidden
            ]).mean()
            future = torch.stack([
                F.cross_entropy(
                    F.linear(projection(norm(hidden.float())), weight).flatten(0, 1),
                    future_targets[index].flatten())
                for index, (norm, projection) in enumerate(
                    zip(self.future_norms, self.future_projections))
            ]).mean()
            total = primary + self.deep_supervision_weight * deep + self.future_prediction_weight * future
        return total, {"primary": primary.detach(), "deep": deep.detach(), "future": future.detach()}

    def forward(self, ids):
        weight = self.composed_weight()
        hidden = self._features_with_weight(ids, weight)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self._prefix_log_probs(hidden.float(), ids, weight.float())

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Multi-token parent source changed")
    return ByteComposedLM(config)
