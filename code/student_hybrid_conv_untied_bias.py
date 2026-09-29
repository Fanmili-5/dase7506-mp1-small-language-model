"""Hybrid-conv inference model with independent output weights and bias."""
from __future__ import annotations

import hashlib
from pathlib import Path

from torch import nn

import student_hybrid_conv_output_bias

PARENT_SHA = "8f2144117213131bf1d621809ce3c520f932ec66f0ffb03796da81bd13a2b36e"


class HybridConvUntiedBiasLM(
    student_hybrid_conv_output_bias.HybridConvOutputBiasLM
):
    def __init__(self, config: dict):
        if config.get("untied_output") is not True:
            raise ValueError("Expected untied_output=true")
        super().__init__(config)
        self.head = nn.Linear(int(config["width"]), self.vocab, bias=False)


def build_model(config: dict) -> HybridConvUntiedBiasLM:
    actual = hashlib.sha256(
        Path(student_hybrid_conv_output_bias.__file__).read_bytes()
    ).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Output-bias parent source changed")
    return HybridConvUntiedBiasLM(config)
