"""Same fixed hybrid predictor, with a collapsed sparse backoff recurrence.

Only call-local prefix queries and immutable training-count buffers are used.
Historical model modules remain unchanged; training keeps the old recurrence.
"""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
import student_ngram_fast

PARENT_SHA = "fda4b7b04bc91cfee6466c13fe610525bdcc0d41fa17a4277e271873237b6fd0"


class CollapsedNgramLM(student_ngram_fast.FastNgramLM):
    def add_into(self, result, ids, scale):
        """Add scale*q(prefix) to a caller-owned dense probability tensor.

        q_i = backoff_i*q_(i-1) + sparse_mass_i expands to one unigram
        term plus sparse terms scaled by the product of higher backoffs.
        No per-order full-vocabulary multiplication or dense count output.
        """
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent nonempty windows <=256")
        batch, length = ids.shape
        if result.shape != (batch, length, self.vocab) or not result.is_contiguous():
            raise ValueError("Expected contiguous caller-owned full distribution")
        flat = result.view(-1, self.vocab)
        positions = torch.arange(length, device=ids.device).expand(batch, -1).flatten()
        query_rows = torch.arange(ids.numel(), device=ids.device)
        suffix_scale = torch.full((ids.numel(),), scale, device=ids.device, dtype=torch.float32)
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
            locations = torch.searchsorted(table.keys, keys).clamp_max(table.keys.numel() - 1)
            found = (table.keys[locations] == keys) & (positions >= history - 1)
            sizes = (table.offsets[locations + 1] - table.offsets[locations]) * found
            rows = torch.repeat_interleave(query_rows, sizes)
            starts = torch.repeat_interleave(table.offsets[locations], sizes)
            local = torch.arange(rows.numel(), device=ids.device) - torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            edges = starts + local
            terms.append((rows, table.values[edges].long(), table.mass[edges] * suffix_scale[rows]))
            suffix_scale.mul_(torch.where(found, table.backoff[locations], 1.0))
        flat.addcmul_(suffix_scale[:, None], self.unigram[None, :])
        for rows, columns, mass in reversed(terms):
            # Unique columns per CSR row; orders are applied separately.
            # Concatenating orders into advanced indexing would lose collisions.
            flat[rows, columns] = flat[rows, columns] + mass
        return result


class CollapsedHybridLM(student_ngram_fast.FastHybridLM):
    def __init__(self, config):
        super().__init__(config)
        self.ngram = CollapsedNgramLM(config)

    def forward(self, ids):
        if self.training:
            return super().forward(ids)
        hidden = self.neural.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            result = F.softmax(self.neural.head(hidden), dim=-1)
            copy = self.neural.copy_distribution(hidden, ids)
            gate = self.neural.copy_gate(hidden)
            result.mul_(torch.sigmoid(-gate) * (1 - self.weight))
            result.addcmul_(copy, torch.sigmoid(gate) * (1 - self.weight))
            self.ngram.add_into(result, ids, self.weight)
            result.div_(result.sum(dim=-1, keepdim=True))
            return result.log()


def build_model(config):
    if hashlib.sha256(Path(student_ngram_fast.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage20 implementation changed")
    # Validate the complete original import chain and supported configuration.
    if hashlib.sha256(Path(student_ngram_fast.student_ngram.__file__).read_bytes()).hexdigest() != student_ngram_fast.PARENT_SHA:
        raise ValueError("Original count-model source changed")
    if config.get("kind") != "hybrid":
        raise ValueError("This module exports hybrid inference only")
    return CollapsedHybridLM(config)
