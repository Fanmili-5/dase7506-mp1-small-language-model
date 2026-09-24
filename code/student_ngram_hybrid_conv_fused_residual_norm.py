"""Fused-copy hybrid that directly logs the normalized convex mixture."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_ngram_hybrid_conv_fused_copy_collapsed


class FusedResidualNormLM(
        student_ngram_hybrid_conv_fused_copy_collapsed.FusedCopyOutputBiasCollapsedLM):
    """Omit a redundant dense division; bound FP32 row drift at export."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(
                self.neural.head(hidden) + self.neural.output_bias, dim=-1)

            raw_gate = self.neural.copy_gate(hidden)
            vocabulary_scale = torch.sigmoid(-raw_gate) * (1 - self.weight)
            copy_scale = (1 - self.weight) - vocabulary_scale
            result.mul_(vocabulary_scale)

            query = self.neural.copy_query(hidden).float()
            key = self.neural.copy_key(hidden).float()
            scores = (query @ key.transpose(-1, -2)) * self.neural.copy_scale
            length = ids.shape[1]
            attention = scores.masked_fill(
                self.copy_future_mask[:length, :length], float("-inf")
            ).softmax(-1)
            attention.mul_(copy_scale)
            result.scatter_add_(
                -1,
                ids[:, None, :].expand(-1, length, -1),
                attention,
            )

            self.ngram.add_into(result, ids, self.weight)
            return result.log()

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> FusedResidualNormLM:
    if config.get("kind") != "hybrid":
        raise ValueError("This module exports hybrid inference only")
    return FusedResidualNormLM(config)
