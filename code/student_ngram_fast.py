"""Equivalent FP32 inference for the existing fixed-weight count/copy hybrid.

No learned tensor changes, extra data, persistent prefix state or target inputs.
Training uses the original log-space neural head. Eval fuses both mixtures.
"""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
import student_ngram

PARENT_SHA = "0b04b2a8c2869679a770176e71b45b970a9dcccd2a941b39f9d04655e538ebf2"


class FastNgramLM(student_ngram.NgramLM):
    def distribution(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent nonempty windows <=256")
        batch, length = ids.shape
        result = self.unigram.expand(batch * length, -1).clone()
        positions = torch.arange(length, device=ids.device).expand(batch, -1)
        for order, table in enumerate(self.tables, 2):
            history = order - 1
            if history > length or table.keys.numel() == 0:
                continue
            keys = torch.zeros_like(ids)
            for lag in range(history - 1, -1, -1):
                shifted = ids if lag == 0 else F.pad(ids[:, :-lag], (lag, 0))
                keys = keys * 2048 + shifted
            keys = keys.flatten()
            locations = torch.searchsorted(table.keys, keys).clamp_max(table.keys.numel() - 1)
            found = (table.keys[locations] == keys) & (positions.flatten() >= history - 1)
            weights = torch.where(found, table.backoff[locations], 1.0)
            result.mul_(weights[:, None])
            sizes = (table.offsets[locations + 1] - table.offsets[locations]) * found
            rows = torch.repeat_interleave(torch.arange(keys.numel(), device=ids.device), sizes)
            starts = torch.repeat_interleave(table.offsets[locations], sizes)
            local = torch.arange(rows.numel(), device=ids.device) - torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            edges = starts + local
            columns = table.values[edges].long()
            # CSR contexts have unique next-token IDs, so (row,column) has no
            # duplicates. Mutate only this call's fresh result, never buffers.
            result[rows, columns] = result[rows, columns] + table.mass[edges]
        return result.view(batch, length, self.vocab)


class FastHybridLM(student_ngram.HybridLM):
    def __init__(self, config):
        super().__init__(config)
        if not 0 < self.weight < 1 or self.neural.output_kind != "prefix_copy":
            raise ValueError("Fused inference requires positive count weight and prefix copy")
        self.ngram = FastNgramLM(config)
        self._check_positive_floor()
        self.register_load_state_dict_post_hook(self._loaded)

    def _loaded(self, module, incompatible_keys):
        self._check_positive_floor()

    def _check_positive_floor(self):
        # Train-count backoff guarantees positive mass for EVERY vocabulary
        # item. Certify once at load, not with an expensive per-batch scan.
        unigram = self.ngram.unigram
        if not torch.isfinite(unigram).all() or unigram.min() <= 0:
            raise ValueError("Unigram must have finite positive support")
        bound = self.weight * float(unigram.min())
        for table in self.ngram.tables:
            if not table.backoff.numel():
                continue
            if (not torch.isfinite(table.backoff).all() or table.backoff.min() <= 0
                    or table.backoff.max() > 1 or not torch.isfinite(table.mass).all()
                    or (table.mass < 0).any()):
                raise ValueError("Invalid positive-backoff statistics")
            bound *= float(table.backoff.min())
        if bound <= 16 * torch.finfo(torch.float32).tiny:
            raise ValueError("Count floor too small for safe FP32 probability-space mixture")
        self.count_probability_lower_bound = bound

    def forward(self, ids):
        if self.training:
            return super().forward(ids)
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.softmax(self.neural.head(hidden), dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden)
            # sigmoid(-gate) avoids cancellation in 1-sigmoid(gate).
            vocabulary.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            copy.mul_(torch.sigmoid(gate) * (1 - self.weight))
            result = self.ngram.distribution(ids)
            result.mul_(self.weight).add_(vocabulary).add_(copy)
            # Correct accumulated FP32 summation drift before the final log.
            # Mathematically the mixture already sums to one.
            result.div_(result.sum(dim=-1, keepdim=True))
            # Certified positive train-count floor prevents log(0), even for
            # extreme finite neural logits. No probability clipping is used.
            return result.log()


def build_model(config):
    if hashlib.sha256(Path(student_ngram.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Original count-model source changed")
    if config.get("kind") != "hybrid":
        raise ValueError("This module exports hybrid inference only")
    return FastHybridLM(config)
