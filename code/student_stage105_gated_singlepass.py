"""Single-pass sparse lookup for gated order-6 MKN inference."""
from __future__ import annotations

import torch
from torch.nn import functional as F

import student_stage104_gated_fast


class SinglePassNgramLM(student_stage104_gated_fast.DynamicCollapsedNgramLM):
    def collect(self, ids):
        """Collect sparse terms and causal confidence in the same table traversal."""
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent causal windows")
        batch, length = ids.shape
        rows = torch.arange(ids.numel(), device=ids.device)
        positions = torch.arange(length, device=ids.device).expand(batch, -1).flatten()
        suffix = torch.ones(ids.numel(), device=ids.device)
        highest_found = torch.zeros(ids.numel(), device=ids.device, dtype=torch.bool)
        highest_backoff = torch.ones(ids.numel(), device=ids.device)
        highest_max_mass = torch.zeros(ids.numel(), device=ids.device)
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
            repeated_rows = torch.repeat_interleave(rows, sizes)
            starts = torch.repeat_interleave(table.offsets[locations], sizes)
            local = torch.arange(repeated_rows.numel(), device=ids.device)
            local -= torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            edges = starts + local
            mass = table.mass[edges]
            terms.append((repeated_rows, table.values[edges].long(),
                          mass * suffix[repeated_rows]))
            new_highest = found & ~highest_found
            if new_highest.any():
                maxima = torch.zeros(ids.numel(), device=ids.device)
                maxima.scatter_reduce_(0, repeated_rows, mass, reduce="amax",
                                       include_self=True)
                highest_backoff[new_highest] = table.backoff[locations[new_highest]]
                highest_max_mass[new_highest] = maxima[new_highest]
                highest_found |= new_highest
            suffix.mul_(torch.where(found, table.backoff[locations], 1.0))
        return (terms, suffix, highest_backoff.reshape(batch, length),
                highest_max_mass.reshape(batch, length))

    def add_collected(self, result, weight, terms, suffix):
        if result.shape[-1] != self.vocab or weight.shape != result.shape[:2]:
            raise ValueError("Invalid collected count tensors")
        flat = result.view(-1, self.vocab)
        flat.addcmul_((suffix * weight.flatten())[:, None], self.unigram[None, :])
        flat_weight = weight.flatten()
        for rows, columns, mass in reversed(terms):
            flat[rows, columns] = flat[rows, columns] + mass * flat_weight[rows]
        return result


class SinglePassGatedLM(student_stage104_gated_fast.FastGatedHybridLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ngram = SinglePassNgramLM(config)

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


def build_model(config: dict) -> SinglePassGatedLM:
    return SinglePassGatedLM(config)
