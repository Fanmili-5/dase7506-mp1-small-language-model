"""Pruned train-only interpolated absolute-discount n-grams and neural mixture.

Produces the complete vocabulary distribution; never receives targets. Immutable
CSR tables are checkpoint buffers. Only the input prefix is used for queries.
Standalone counts are diagnostic; the submission candidate is a trainable neural
model plus these training-derived statistics, not a dummy-gradient lookup model.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn


class OrderTable(nn.Module):
    def __init__(self, contexts, edges):
        super().__init__()
        self.register_buffer("keys", torch.zeros(contexts, dtype=torch.int64))
        self.register_buffer("offsets", torch.zeros(contexts + 1, dtype=torch.int64))
        self.register_buffer("values", torch.zeros(edges, dtype=torch.int32))
        self.register_buffer("mass", torch.zeros(edges, dtype=torch.float32))
        self.register_buffer("backoff", torch.ones(contexts, dtype=torch.float32))


class NgramLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context, self.vocab = int(config["context"]), int(config["vocab"])
        if (self.context, self.vocab) != (256, 2048):
            raise ValueError("Fixed context/vocabulary required")
        self.register_buffer("unigram", torch.full((2048,), 1 / 2048))
        self.tables = nn.ModuleList([OrderTable(*shape) for shape in config["order_shapes"]])

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
            # Encode exactly the suffix x_(t-history+1),...,x_t; no future input.
            keys = torch.zeros_like(ids)
            for lag in range(history - 1, -1, -1):
                shifted = ids if lag == 0 else torch.nn.functional.pad(ids[:, :-lag], (lag, 0))
                keys = keys * 2048 + shifted
            keys = keys.flatten()
            locations = torch.searchsorted(table.keys, keys).clamp_max(table.keys.numel() - 1)
            found = (table.keys[locations] == keys) & (positions.flatten() >= history - 1)
            weights = torch.where(found, table.backoff[locations], 1.0)
            result = result * weights[:, None]
            sizes = (table.offsets[locations + 1] - table.offsets[locations]) * found
            rows = torch.repeat_interleave(torch.arange(keys.numel(), device=ids.device), sizes)
            starts = torch.repeat_interleave(table.offsets[locations], sizes)
            local = torch.arange(rows.numel(), device=ids.device) - torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
            edges = starts + local
            # Each CSR row has unique next-token IDs. Different query rows cannot
            # collide. Out-of-place index_put preserves autograd for hybrid use.
            increments = torch.zeros_like(result)
            increments.index_put_((rows, table.values[edges].long()), table.mass[edges])
            result = result + increments
        return result.view(batch, length, self.vocab)

    def forward(self, ids):
        return self.distribution(ids).log()

    def predict_log_probs(self, ids):
        return self(ids)


class HybridLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        import student
        import student_structured
        expected = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"
        if hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("Structured implementation changed")
        self.neural = student_structured.build_model(config["neural_config"])
        self.ngram = NgramLM(config)
        self.context, self.vocab = 256, 2048
        self.weight = float(config["mixture_weight"])
        if not 0 <= self.weight < 1:
            raise ValueError("Mixture must retain a neural component")

    def forward(self, ids):
        neural = self.neural.predict_log_probs(ids)
        if self.weight == 0:
            return neural
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return torch.logaddexp(neural.float() + math.log1p(-self.weight),
                                   self.ngram(ids) + math.log(self.weight))

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    return HybridLM(config) if config.get("kind") == "hybrid" else NgramLM(config)
