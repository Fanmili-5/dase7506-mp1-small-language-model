"""Stage186 inference view without training-only auxiliary heads."""
from __future__ import annotations

import student_hybrid_conv_structured
from student_stage186_narrow_parallel import install_narrow_branches


class NarrowParallelStructuredLM(student_hybrid_conv_structured.HybridConvStructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_narrow_branches(self, config)


def build_model(config: dict) -> NarrowParallelStructuredLM:
    return NarrowParallelStructuredLM(config)
