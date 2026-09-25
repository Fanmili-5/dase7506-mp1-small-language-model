"""Training-only loss emphasis; inference is the unchanged hybrid model."""
from __future__ import annotations

import torch

import student_hybrid_conv_rdrop

FOCUS_ALPHA = 0.25
MIN_TRAIN_FREQUENCY = 100
MAX_TRAIN_FREQUENCY = 999


def unseen_medium_frequency_mask(ids: torch.Tensor, targets: torch.Tensor,
                                 train_frequency: torch.Tensor) -> torch.Tensor:
    """Select training targets absent from their independent causal prefix."""
    if ids.ndim != 2 or ids.shape != targets.shape or train_frequency.shape != (2048,):
        raise ValueError("Expected [batch,time] token IDs and 2,048 counts")
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device)
    preceding = positions[None, None, :] <= positions[None, :, None]
    seen = ((ids[:, None, :] == targets[:, :, None]) & preceding).any(-1)
    counts = train_frequency[targets]
    medium = (counts >= MIN_TRAIN_FREQUENCY) & (counts <= MAX_TRAIN_FREQUENCY)
    return medium & ~seen


def normalized_focused_primary(logp: torch.Tensor, targets: torch.Tensor,
                               indicator: torch.Tensor, alpha: float) -> torch.Tensor:
    if (logp.shape != (*targets.shape, 2048) or indicator.shape != targets.shape
            or not 0 <= alpha <= 1):
        raise ValueError("Invalid focused-loss shapes or weight")
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)
    weights = 1.0 + alpha * indicator.to(nll.dtype)
    return (nll * weights).sum() / weights.sum()


class FrequencyFocusLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    """Identical model tensors; only training main-loss weighting differs."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.register_buffer("train_frequency", torch.zeros(2048, dtype=torch.long),
                             persistent=False)
        self.frequency_ready = False

    def set_train_frequency(self, counts: torch.Tensor) -> None:
        if counts.shape != (2048,) or counts.dtype != torch.long or (counts < 0).any():
            raise ValueError("Expected nonnegative 2,048-way train counts")
        self.train_frequency.copy_(counts.to(self.train_frequency.device))
        self.frequency_ready = True

    def single_training_pass(self, ids, targets, future_targets):
        if not self.frequency_ready:
            raise ValueError("Set training frequencies before Stage161 training")
        total, logp, parts = super().single_training_pass(
            ids, targets, future_targets)
        indicator = unseen_medium_frequency_mask(ids, targets, self.train_frequency)
        primary = normalized_focused_primary(logp, targets, indicator, FOCUS_ALPHA)
        ordinary = -logp.gather(-1, targets[..., None]).squeeze(-1).mean()
        # The original main loss contributes gradient through `total`; replace
        # it with a normalized focused loss without changing other objectives.
        total = total + primary - ordinary
        parts["primary"] = primary.detach()
        parts["focus_fraction"] = indicator.float().mean().detach()
        return total, logp, parts


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> FrequencyFocusLM:
    return FrequencyFocusLM(config)
