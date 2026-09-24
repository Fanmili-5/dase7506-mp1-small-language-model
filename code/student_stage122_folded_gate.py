"""Five-order dynamic gate with folded fixed calibration and cached count maxima."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_stage116_rowmax_gate


class FoldedGateLM(student_stage116_rowmax_gate.RowMaxGatedLM):
    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent causal windows")
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(
                self.neural.head(hidden) + self.neural.output_bias, dim=-1)
            gate = self.neural.copy_gate(hidden)
            vocabulary_scale = torch.sigmoid(-gate)
            copy_scale = 1 - vocabulary_scale
            result.mul_(vocabulary_scale)
            query = self.neural.copy_query(hidden).float()
            key = self.neural.copy_key(hidden).float()
            scores = (query @ key.transpose(-1, -2)) * self.neural.copy_scale
            length = ids.shape[1]
            attention = scores.masked_fill(
                self.copy_future_mask[:length, :length], float("-inf")).softmax(-1)
            attention.mul_(copy_scale)
            result.scatter_add_(-1, ids[:, None, :].expand(-1, length, -1), attention)
            top = result.topk(2, dim=-1).values.log()
            terms, suffix, backoff, max_mass = self.ngram.collect(ids)
            features = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                                    backoff, max_mass), dim=-1).double()
            slope = (features * self.gate_alpha).sum(-1) + self.gate_intercept
            weight = torch.sigmoid(self.slope_scale * slope + self.anchor_logit)
            weight = (1e-4 + (1 - 2e-4) * weight).float()
            result.mul_((1 - weight).unsqueeze(-1))
            self.ngram.add_collected(result, weight, terms, suffix)
            return result.log()


def build_model(config):
    return FoldedGateLM(config)
