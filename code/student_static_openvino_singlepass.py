"""Same frozen FP32 graph/heads/counts, executed in fixed eight-row blocks.

Only independent batch rows are split. Right padding is causal and discarded;
there is no cross-call token state, output cache, quantization or new parameter.
"""
from __future__ import annotations

import hashlib

import numpy as np
import openvino as ov
import torch
from torch import nn

import student_stage105_gated_singlepass
from student_stage143_openvino_singlepass import GRAPH, GRAPH_SHA256

STATIC_BATCH = 8


class StaticGraphNeuralHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width, vocab, copy_dim = (int(config[k]) for k in ('width', 'vocab', 'copy_dim'))
        if (width, vocab, copy_dim, int(config['context'])) != (288, 2048, 64, 256):
            raise ValueError('Unexpected frozen feature/head configuration')
        self.head = nn.Linear(width, vocab, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab))
        self.copy_query = nn.Linear(width, copy_dim, bias=False)
        self.copy_key = nn.Linear(width, copy_dim, bias=False)
        self.copy_gate = nn.Linear(width, 1)
        self.copy_scale = copy_dim ** -0.5
        if not GRAPH.is_file() or hashlib.sha256(GRAPH.read_bytes()).hexdigest() != GRAPH_SHA256:
            raise ValueError('Missing or changed frozen FP32 graph')
        core = ov.Core()
        graph = core.read_model(str(GRAPH))
        graph.reshape({'ids': [STATIC_BATCH, 256]})
        requested = torch.get_num_threads()
        self._compiled = core.compile_model(graph, 'CPU', {
            'INFERENCE_PRECISION_HINT': ov.Type.f32,
            'INFERENCE_NUM_THREADS': requested,
            'NUM_STREAMS': 1,
            'ENABLE_CPU_PINNING': False,
            # Permit logical workers, still capped by the requested budget.
            'ENABLE_HYPER_THREADING': True,
        })
        used = int(self._compiled.get_property('INFERENCE_NUM_THREADS'))
        if (self._compiled.get_property('INFERENCE_PRECISION_HINT') != ov.Type.f32
                or not 1 <= used <= requested
                or int(self._compiled.get_property('NUM_STREAMS')) != 1
                or self._compiled.get_property('ENABLE_CPU_PINNING')):
            raise ValueError('CPU precision/thread budget/stream settings changed')
        self.execution_threads = used
        self._request = self._compiled.create_infer_request()

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        if (ids.device.type != 'cpu' or ids.dtype != torch.long or ids.ndim != 2
                or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= 256):
            raise ValueError('Expected nonempty independent CPU int64 windows')
        length = ids.shape[1]
        outputs = []
        for start in range(0, len(ids), STATIC_BATCH):
            block = ids[start:start + STATIC_BATCH]
            padded = np.zeros((STATIC_BATCH, 256), dtype=np.int64)
            padded[:len(block), :length] = block.numpy()
            values = self._request.infer({'ids': padded})
            hidden = next(iter(values.values()))
            if hidden.shape != (STATIC_BATCH, 256, 288) or hidden.dtype != np.float32:
                raise ValueError('Unexpected frozen graph output')
            # Own the output: a later request must not overwrite previous rows.
            outputs.append(torch.from_numpy(hidden[:len(block), :length].copy()))
        return torch.cat(outputs, dim=0)


class StaticOpenVinoGatedLM(student_stage105_gated_singlepass.SinglePassGatedLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.neural = StaticGraphNeuralHead(config['neural_config'])


def build_model(config: dict) -> StaticOpenVinoGatedLM:
    return StaticOpenVinoGatedLM(config)
