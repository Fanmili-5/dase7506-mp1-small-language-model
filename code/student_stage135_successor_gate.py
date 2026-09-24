"""Stage115 with one causal successor-copy route instead of content-copy."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_stage115_order5_gate


def add_successor_copy(result: torch.Tensor, ids: torch.Tensor,
                       scores: torch.Tensor, copy_scale: torch.Tensor) -> torch.Tensor:
    """Add attention over earlier contexts' observed successors in this row."""
    batch, length = ids.shape
    if (result.shape[:2] != ids.shape or scores.shape != (batch, length, length)
            or copy_scale.shape != (batch, length, 1)):
        raise ValueError("Incompatible successor-copy shapes")
    if length == 1:
        return result
    # At t>=1 only j<t is visible. Row zero is ignored and never scattered.
    blocked = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(0)
    blocked[0, 0] = False  # avoid an all-masked softmax row
    attention = scores.masked_fill(blocked, float("-inf")).softmax(-1)
    attention = attention[:, 1:, :] * copy_scale[:, 1:, :]
    successors = torch.cat((ids[:, 1:], ids[:, -1:]), dim=-1)
    result[:, 1:, :].scatter_add_(
        -1, successors[:, None, :].expand(-1, length - 1, -1), attention)
    return result


class SuccessorGatedLM(student_stage115_order5_gate.Order5GatedLM):
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent causal windows")
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(
                (self.neural.head(hidden) + self.neural.output_bias)
                / self.temperature + self.prior_weight * self.log_prior,
                dim=-1)
            gate = self.neural.copy_gate(hidden) + self.copy_shift
            vocabulary_scale = torch.sigmoid(-gate)
            first = torch.arange(ids.shape[1], device=ids.device).eq(0)[None, :, None]
            vocabulary_scale = torch.where(first, 1., vocabulary_scale)
            copy_scale = 1 - vocabulary_scale
            result.mul_(vocabulary_scale)
            query = self.neural.copy_query(hidden).float()
            key = self.neural.copy_key(hidden).float()
            scores = (query @ key.transpose(-1, -2)) * self.neural.copy_scale
            add_successor_copy(result, ids, scores, copy_scale)
            top = result.topk(2, dim=-1).values.log()
            terms, suffix, backoff, max_mass = self.ngram.collect(ids)
            features = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                                    backoff, max_mass), dim=-1).double()
            slope = ((features - self.gate_mean) / self.gate_std)
            slope = (slope * self.gate_coeff).sum(-1)
            weight = torch.sigmoid(self.anchor_logit + self.slope_scale * slope)
            weight = (1e-4 + (1 - 2e-4) * weight).float()
            result.mul_((1 - weight).unsqueeze(-1))
            self.ngram.add_collected(result, weight, terms, suffix)
            return result.log()


def build_model(config: dict) -> SuccessorGatedLM:
    return SuccessorGatedLM(config)
