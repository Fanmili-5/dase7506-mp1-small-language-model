"""Stage186 R-Drop training view with four narrow global branches."""
from __future__ import annotations

import student_hybrid_conv_rdrop
from student_stage186_narrow_parallel import install_narrow_branches


class NarrowParallelRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_narrow_branches(self, config)


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> NarrowParallelRDropLM:
    return NarrowParallelRDropLM(config)
