"""Evaluation model for the Stage223 bigram-input architecture."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_structured
import student_stage223_bigram_input
from student_stage223_bigram_input import BigramInputMixin


PARENT_SHA = "81e61cc36bc6aec3711de27c9860c7ac6b80238b8b5481c5bfa7710a04d9c34e"
CORE_SHA = "60dc2dfd03343e7aedaf80781dd2a787f3a9995856999759fd55985b1dbe47c9"


class BigramStructuredLM(BigramInputMixin, student_hybrid_conv_structured.HybridConvStructuredLM):
    pass


def build_model(config: dict) -> BigramStructuredLM:
    if hashlib.sha256(Path(student_hybrid_conv_structured.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 inference source changed")
    if hashlib.sha256(Path(student_stage223_bigram_input.__file__).read_bytes()).hexdigest() != CORE_SHA:
        raise ValueError("Stage223 bigram core source changed")
    return BigramStructuredLM(config)
