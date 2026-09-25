"""Stage177 inference view without train-only R-Drop or auxiliary heads."""
from __future__ import annotations

import student_hybrid_conv_structured
from student_stage177_parallel_mixer import install_parallel_branches


class ParallelMixerStructuredLM(student_hybrid_conv_structured.HybridConvStructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_parallel_branches(self, config)


def build_model(config: dict) -> ParallelMixerStructuredLM:
    return ParallelMixerStructuredLM(config)
