"""Training-only deep/future supervision for the top-1 routed Transformer."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

import student_moe_structured
import student_multi_token

PARENT_SHA = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"
MOE_TRAINING_KEYS = ("moe_balance_weight", "moe_z_weight")


class MoEMultiTokenLM(student_multi_token.MultiTokenLM):
    def __init__(self, config: dict):
        super().__init__(config)
        student_moe_structured.install_moe_layers(self, config)
        self.moe_balance_weight = float(config.get("moe_balance_weight", 0.01))
        self.moe_z_weight = float(config.get("moe_z_weight", 0.001))
        if not math.isfinite(self.moe_balance_weight) or self.moe_balance_weight < 0:
            raise ValueError("moe_balance_weight must be finite and nonnegative")
        if not math.isfinite(self.moe_z_weight) or self.moe_z_weight < 0:
            raise ValueError("moe_z_weight must be finite and nonnegative")

    def training_loss(self, ids, targets, future_targets):
        base_loss, parts = super().training_loss(ids, targets, future_targets)
        routed = [self.blocks[layer - 1].mlp for layer in self.moe_layers]
        if any(module.router_balance_loss is None for module in routed):
            raise RuntimeError("Missing router statistics from the training forward pass")
        balance = torch.stack([module.router_balance_loss for module in routed]).mean()
        z_loss = torch.stack([module.router_z_loss for module in routed]).mean()
        total = base_loss + self.moe_balance_weight * balance + self.moe_z_weight * z_loss
        return total, parts | {"router_balance": balance.detach(), "router_z": z_loss.detach()}


def inference_config(config: dict) -> dict:
    result = student_multi_token.inference_config(config)
    for key in MOE_TRAINING_KEYS:
        result.pop(key, None)
    return result


def inference_state(state: dict) -> dict:
    return student_multi_token.inference_state(state)


def build_model(config: dict) -> MoEMultiTokenLM:
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Multi-token parent source changed")
    return MoEMultiTokenLM(config)
