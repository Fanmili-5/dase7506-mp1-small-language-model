"""Single-copy FP32 OpenVINO feature graph with frozen Stage105 heads and MKN."""
from __future__ import annotations

import hashlib
from pathlib import Path

import openvino as ov
import torch
from torch import nn

import student_stage105_gated_singlepass


GRAPH_SHA256 = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
GRAPH = Path(__file__).resolve().parent / "inference_assets/stage143-stage92-features.onnx"


class FrozenGraphNeuralHead(nn.Module):
    """Store only the output/copy head; feature weights live once in ONNX."""

    def __init__(self, config: dict):
        super().__init__()
        width = int(config["width"])
        vocab = int(config["vocab"])
        copy_dim = int(config["copy_dim"])
        if (width != 288 or vocab != 2048 or copy_dim != 64
                or int(config["context"]) != 256):
            raise ValueError("Unexpected frozen Stage92 feature/head configuration")
        self.head = nn.Linear(width, vocab, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab))
        self.copy_query = nn.Linear(width, copy_dim, bias=False)
        self.copy_key = nn.Linear(width, copy_dim, bias=False)
        self.copy_gate = nn.Linear(width, 1)
        self.copy_scale = copy_dim ** -0.5
        if (not GRAPH.is_file()
                or hashlib.sha256(GRAPH.read_bytes()).hexdigest() != GRAPH_SHA256):
            raise ValueError("Missing or changed frozen FP32 feature graph")
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
        reported = self._compiled.get_property("INFERENCE_PRECISION_HINT")
        used_threads = self._compiled.get_property("INFERENCE_NUM_THREADS")
        streams = self._compiled.get_property("NUM_STREAMS")
        pinning = self._compiled.get_property("ENABLE_CPU_PINNING")
        if (reported != ov.Type.f32 or int(used_threads) != threads
                or int(streams) != 1 or pinning):
            raise ValueError("OpenVINO FP32/thread/stream settings changed")

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        if (ids.device.type != "cpu" or ids.ndim != 2
                or not 1 <= ids.shape[1] <= 256):
            raise ValueError("Stage143 accepts only independent CPU windows")
        if ids.shape[1] != 256:
            raise ValueError("Frozen feature graph requires padded 256-token windows")
        output = self._compiled({"ids": ids.numpy()})
        hidden = torch.from_numpy(next(iter(output.values())))
        if hidden.shape != (*ids.shape, 288) or hidden.dtype != torch.float32:
            raise ValueError("Invalid frozen feature graph output")
        return hidden


class OpenVinoSinglePassGatedLM(
        student_stage105_gated_singlepass.SinglePassGatedLM):
    def __init__(self, config: dict):
        super().__init__(config)
        # The inherited constructor validates the complete frozen inference
        # contract; discard its temporary PyTorch backbone before loading.
        self.neural = FrozenGraphNeuralHead(config["neural_config"])


def build_model(config: dict) -> OpenVinoSinglePassGatedLM:
    return OpenVinoSinglePassGatedLM(config)
