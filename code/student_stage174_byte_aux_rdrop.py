"""Stage54 hybrid model with next-token ByteLevel supervision only at training."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_byte_composed
import student_hybrid_conv_rdrop


PARENT_SHA256 = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
BYTE_SOURCE_SHA256 = "59ae25c00df3d2b46fa20b20feff2e31af267fc289cb1dd1ce84b65bcaf9010b"
AUX_WEIGHT_KEY = "token_byte_aux_weight"


class ByteAuxHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        weight = float(config[AUX_WEIGHT_KEY])
        if not math.isfinite(weight) or weight != 0.2:
            raise ValueError("Stage174 fixes token_byte_aux_weight=0.2")
        self.byte_aux_weight = weight
        width = int(config["width"])
        self.aux_first = nn.Linear(width, 256)
        self.aux_last = nn.Linear(width, 256)
        features = student_byte_composed.build_byte_features(self.vocab)
        first = features[:, 256:512]
        last = features[:, 512:768]
        if not torch.equal(first.sum(-1), torch.ones(self.vocab)) or not torch.equal(
                last.sum(-1), torch.ones(self.vocab)):
            raise ValueError("Each token must have exactly one first/last byte")
        self.register_buffer("first_byte", first.argmax(-1), persistent=False)
        self.register_buffer("last_byte", last.argmax(-1), persistent=False)

    def rdrop_training_loss(self, ids, targets, future_targets):
        if not self.training:
            raise ValueError("Byte auxiliary loss is training-only")
        captured = []
        handle = self.norm.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output))
        try:
            ordinary_loss, parts = super().rdrop_training_loss(ids, targets, future_targets)
        finally:
            handle.remove()
        if len(captured) != 2 or any(hidden.shape != (*ids.shape, self.config["width"])
                                      for hidden in captured):
            raise ValueError("Expected one causal final hidden state per R-Drop pass")
        first_target = self.first_byte[targets].reshape(-1)
        last_target = self.last_byte[targets].reshape(-1)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            losses = []
            for hidden in captured:
                first_logits = self.aux_first(hidden.float()).float().reshape(-1, 256)
                last_logits = self.aux_last(hidden.float()).float().reshape(-1, 256)
                losses.append(0.5 * (
                    F.cross_entropy(first_logits, first_target)
                    + F.cross_entropy(last_logits, last_target)))
            byte_loss = torch.stack(losses).mean()
            total = ordinary_loss + self.byte_aux_weight * byte_loss
        return total, dict(parts, byte_aux=byte_loss.detach())


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(
        {key: value for key, value in config.items() if key != AUX_WEIGHT_KEY})


def inference_state(state: dict) -> dict:
    stripped = {key: value for key, value in state.items()
                if not key.startswith(("aux_first.", "aux_last."))}
    return student_hybrid_conv_rdrop.inference_state(stripped)


def build_model(config: dict) -> ByteAuxHybridLM:
    if hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("Stage54 parent changed")
    if hashlib.sha256(Path(student_byte_composed.__file__).read_bytes()).hexdigest() != BYTE_SOURCE_SHA256:
        raise ValueError("ByteLevel mapping source changed")
    return ByteAuxHybridLM(config)
