"""Seven-block Stage92 architecture with original block five removed."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_output_bias


PARENT_SHA = "8f2144117213131bf1d621809ce3c520f932ec66f0ffb03796da81bd13a2b36e"


class PrunedHybridLM(student_hybrid_conv_output_bias.HybridConvOutputBiasLM):
    def __init__(self, config: dict):
        if int(config["depth"]) != 8:
            raise ValueError("Expected Stage92 eight-block source configuration")
        super().__init__(config)
        del self.blocks[4]


def build_model(config: dict) -> PrunedHybridLM:
    actual = hashlib.sha256(
        Path(student_hybrid_conv_output_bias.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Pinned Stage92 parent source changed")
    return PrunedHybridLM(config)
