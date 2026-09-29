"""Training view of a ten-logical-block, eight-unique-block hybrid LM."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_rdrop
from student_stage221_tied_depth import tie_tail_pair

PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


class TiedDepthRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        tie_tail_pair(self, config)


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> TiedDepthRDropLM:
    if hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage54 hybrid R-Drop parent changed")
    return TiedDepthRDropLM(config)
