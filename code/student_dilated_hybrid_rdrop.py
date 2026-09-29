"""Training view of Stage164's dilated causal-convolution backbone."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_rdrop
import student_dilated_hybrid_structured

PARENT_SHA256 = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"


class DilatedHybridRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        student_dilated_hybrid_structured.install_dilations(self, config)


def inference_config(config: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> DilatedHybridRDropLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Pinned hybrid training parent changed")
    return DilatedHybridRDropLM(config)
