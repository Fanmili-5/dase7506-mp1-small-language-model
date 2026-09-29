"""Inference view of the Stage212 alternating global/local model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_structured
import student_stage212_sliding_local

PARENT_SHA = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class SlidingLocalStructuredLM(student_structured.StructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        student_stage212_sliding_local.install_local_attention_layers(self, config)


def build_model(config: dict) -> SlidingLocalStructuredLM:
    actual = hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Structured parent source changed")
    return SlidingLocalStructuredLM(config)
