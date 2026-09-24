"""Fused-copy and collapsed-count inference for the Stage103 gated model."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_stage103_gated
from student_ngram_collapsed import CollapsedNgramLM


class DynamicCollapsedNgramLM(CollapsedNgramLM):
    def add_into_dynamic(self, result, ids, weight):
        """Add a normalized count distribution with one weight per prefix."""
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent causal windows")
        batch, length = ids.shape
        if result.shape != (batch, length, self.vocab) or not result.is_contiguous():
            raise ValueError("Expected contiguous caller-owned distribution")
        if weight.shape != ids.shape:
            raise ValueError("Expected one count weight per prefix")
        flat = result.view(-1, self.vocab)
        positions = torch.arange(length, device=ids.device).expand(batch, -1).flatten()
        query_rows = torch.arange(ids.numel(), device=ids.device)
        suffix_scale = weight.flatten().float().clone()
        terms = []
        for history in range(len(self.tables), 0, -1):
            table = self.tables[history - 1]
            if history > length or table.keys.numel() == 0:
                continue
            keys = torch.zeros_like(ids)
            for lag in range(history - 1, -1, -1):
                shifted = ids if lag == 0 else F.pad(ids[:, :-lag], (lag, 0))
                keys = keys * self.vocab + shifted
            keys = keys.flatten()
            locations = torch.searchsorted(table.keys, keys).clamp_max(
                table.keys.numel() - 1)
            found = (table.keys[locations] == keys) & (positions >= history - 1)
            sizes = (table.offsets[locations + 1] - table.offsets[locations]) * found
            rows = torch.repeat_interleave(query_rows, sizes)
            starts = torch.repeat_interleave(table.offsets[locations], sizes)
            local = torch.arange(rows.numel(), device=ids.device)
            local -= torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            edges = starts + local
            terms.append((rows, table.values[edges].long(),
                          table.mass[edges] * suffix_scale[rows]))
            suffix_scale.mul_(torch.where(found, table.backoff[locations], 1.0))
        flat.addcmul_(suffix_scale[:, None], self.unigram[None, :])
        for rows, columns, mass in reversed(terms):
            flat[rows, columns] = flat[rows, columns] + mass
        return result


class FastGatedHybridLM(student_stage103_gated.GatedHybridLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ngram = DynamicCollapsedNgramLM(config)
        future = torch.ones(self.context, self.context, dtype=torch.bool).triu(1)
        self.register_buffer("copy_future_mask", future, persistent=False)

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
            copy_scale = 1 - vocabulary_scale
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
                -1, ids[:, None, :].expand(-1, length, -1), attention)
            top = result.topk(2, dim=-1).values.log()
            backoff, max_mass = self._sparse_confidence(ids)
            features = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                                    backoff, max_mass), dim=-1).double()
            slope = ((features - self.gate_mean) / self.gate_std)
            slope = (slope * self.gate_coeff).sum(-1)
            weight = torch.sigmoid(self.anchor_logit + self.slope_scale * slope)
            weight = 1e-4 + (1 - 2e-4) * weight
            weight = weight.float()
            result.mul_((1 - weight).unsqueeze(-1))
            self.ngram.add_into_dynamic(result, ids, weight)
            return result.log()


def build_model(config: dict) -> FastGatedHybridLM:
    return FastGatedHybridLM(config)
