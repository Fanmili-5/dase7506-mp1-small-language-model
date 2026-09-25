"""Stage177 training view: R-Drop with parallel global/local token mixing."""
from __future__ import annotations

import student_hybrid_conv_rdrop
from student_stage177_parallel_mixer import install_parallel_branches


class ParallelMixerRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_parallel_branches(self, config)


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> ParallelMixerRDropLM:
    return ParallelMixerRDropLM(config)
