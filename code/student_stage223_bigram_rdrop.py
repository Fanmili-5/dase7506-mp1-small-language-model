"""Stage54 training model with the fixed Stage223 bigram input."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_rdrop
import student_stage223_bigram_input
from student_stage223_bigram_input import BigramInputMixin


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
CORE_SHA = "60dc2dfd03343e7aedaf80781dd2a787f3a9995856999759fd55985b1dbe47c9"


class BigramRDropLM(BigramInputMixin, student_hybrid_conv_rdrop.HybridConvRDropLM):
    pass


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> BigramRDropLM:
    if hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 R-Drop source changed")
    if hashlib.sha256(Path(student_stage223_bigram_input.__file__).read_bytes()).hexdigest() != CORE_SHA:
        raise ValueError("Stage223 bigram core source changed")
    return BigramRDropLM(config)
