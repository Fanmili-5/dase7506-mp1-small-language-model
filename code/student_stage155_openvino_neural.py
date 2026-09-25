"""Single-copy FP32 OpenVINO features and trained prefix-copy output head."""
from __future__ import annotations

import hashlib
from pathlib import Path

import openvino as ov
import torch
from torch import nn
from torch.nn import functional as F


GRAPH = Path(__file__).resolve().parent / "inference_assets/stage155-full-features.onnx"


class OpenVinoNeuralLM(nn.Module):
    """Keep feature weights only in ONNX; checkpoint stores trained head once."""

    def __init__(self, config: dict):
        super().__init__()
        if (int(config["width"]) != 320 or int(config["vocab"]) != 2048
                or int(config["context"]) != 256
                or int(config["copy_dim"]) != 64):
            raise ValueError("Unexpected Stage155 frozen model dimensions")
        expected = str(config["feature_graph_sha256"])
        if len(expected) != 64 or not GRAPH.is_file():
            raise ValueError("Missing frozen Stage155 feature graph")
        if hashlib.sha256(GRAPH.read_bytes()).hexdigest() != expected:
            raise ValueError("Stage155 feature graph hash mismatch")
        self.context = 256
        self.vocab = 2048
        self.head = nn.Linear(320, 2048, bias=False)
        self.copy_query = nn.Linear(320, 64, bias=False)
        self.copy_key = nn.Linear(320, 64, bias=False)
        self.copy_gate = nn.Linear(320, 1)
        self.copy_scale = 64 ** -0.5
        self.register_buffer("copy_future_mask", torch.ones(256, 256,
                                                             dtype=torch.bool).triu(1),
                             persistent=False)
        core = ov.Core()
        if "CPU" not in core.available_devices:
            raise RuntimeError("OpenVINO CPU execution unavailable")
        threads = torch.get_num_threads()
        self._compiled = core.compile_model(str(GRAPH), "CPU", {
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
            "INFERENCE_NUM_THREADS": threads,
            "NUM_STREAMS": 1,
            "ENABLE_CPU_PINNING": False,
        })
        if (self._compiled.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32
                or int(self._compiled.get_property("INFERENCE_NUM_THREADS")) != threads
                or int(self._compiled.get_property("NUM_STREAMS")) != 1
                or self._compiled.get_property("ENABLE_CPU_PINNING")):
            raise ValueError("Stage155 OpenVINO FP32/thread settings changed")

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        if (ids.device.type != "cpu" or ids.ndim != 2
                or ids.shape[1] != self.context):
            raise ValueError("Expected independent padded CPU windows [batch,256]")
        result = self._compiled({"ids": ids.numpy()})
        hidden = torch.from_numpy(next(iter(result.values())))
        if hidden.shape != (*ids.shape, 320) or hidden.dtype != torch.float32:
            raise ValueError("Invalid frozen feature graph output")
        return hidden

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        with torch.autocast(device_type="cpu", enabled=False):
            vocabulary = F.log_softmax(self.head(hidden), dim=-1)
            query = self.copy_query(hidden).float()
            key = self.copy_key(hidden).float()
            scores = (query @ key.transpose(-1, -2)) * self.copy_scale
            attention = scores.masked_fill(
                self.copy_future_mask, float("-inf")).softmax(-1)
            copy = hidden.new_zeros(*ids.shape, self.vocab)
            copy.scatter_add_(-1, ids[:, None, :].expand(-1, self.context, -1),
                              attention)
            log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
            log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
            gate = self.copy_gate(hidden)
            return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                                   F.logsigmoid(gate) + log_copy)

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)


def build_model(config: dict) -> OpenVinoNeuralLM:
    return OpenVinoNeuralLM(config)
