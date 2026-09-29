"""Training-only low-rank residual for the hybrid-conv vocabulary projection."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student_hybrid_conv_output_bias

PARENT_SHA = "8f2144117213131bf1d621809ce3c520f932ec66f0ffb03796da81bd13a2b36e"


class HybridConvOutputLoRALM(
    student_hybrid_conv_output_bias.HybridConvOutputBiasLM
):
    def __init__(self, config: dict):
        super().__init__(config)
        rank = int(config["output_lora_rank"])
        alpha = float(config["output_lora_alpha"])
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Output LoRA rank and alpha must be positive")
        width = int(config["width"])
        self.output_lora_a = nn.Parameter(torch.empty(rank, width))
        self.output_lora_b = nn.Parameter(torch.zeros(self.vocab, rank))
        nn.init.normal_(self.output_lora_a, std=.02)
        self.output_lora_scale = alpha / rank

    def output_weight(self):
        return self.head.weight + self.output_lora_scale * (
            self.output_lora_b @ self.output_lora_a
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            hidden = hidden.float()
            vocabulary = F.log_softmax(
                F.linear(hidden, self.output_weight()) + self.output_bias,
                dim=-1,
            )
            copy = self.copy_distribution(hidden, ids)
            log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
            log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
            gate = self.copy_gate(hidden)
            return torch.logaddexp(
                F.logsigmoid(-gate) + vocabulary,
                F.logsigmoid(gate) + log_copy,
            )


def build_model(config: dict) -> HybridConvOutputLoRALM:
    actual = hashlib.sha256(
        Path(student_hybrid_conv_output_bias.__file__).read_bytes()
    ).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Output-bias parent source changed")
    return HybridConvOutputLoRALM(config)
