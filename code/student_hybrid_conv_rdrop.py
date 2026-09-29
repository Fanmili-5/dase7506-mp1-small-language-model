"""R-Drop training view of the alternating attention/causal-conv model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_structured
import student_rdrop_multi_token

PARENT_SHA = "6b282f4e19077349e2cd17c9c48094f784924e447403fd751992b6e3295ec79d"


class HybridConvRDropLM(student_rdrop_multi_token.RDropMultiTokenLM):
    def __init__(self, config: dict):
        super().__init__(config)
        student_hybrid_conv_structured.install_conv_layers(self, config)


def inference_config(config: dict) -> dict:
    return student_rdrop_multi_token.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_rdrop_multi_token.inference_state(state)


def build_model(config: dict) -> HybridConvRDropLM:
    actual = hashlib.sha256(Path(student_rdrop_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("R-Drop parent source changed")
    return HybridConvRDropLM(config)
