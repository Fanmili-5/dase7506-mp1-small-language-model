"""Inference view of the tied-depth hybrid architecture."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_structured
from student_stage221_tied_depth import tie_tail_pair

PARENT_SHA = "81e61cc36bc6aec3711de27c9860c7ac6b80238b8b5481c5bfa7710a04d9c34e"


class TiedDepthStructuredLM(student_hybrid_conv_structured.HybridConvStructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        tie_tail_pair(self, config)


def build_model(config: dict) -> TiedDepthStructuredLM:
    if hashlib.sha256(Path(student_hybrid_conv_structured.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 hybrid structured parent changed")
    return TiedDepthStructuredLM(config)
