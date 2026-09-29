"""Matched training objectives for the alternating attention/conv backbone."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_structured
import student_multi_token

PARENT_SHA = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"


class HybridConvMultiTokenLM(student_multi_token.MultiTokenLM):
    def __init__(self, config: dict):
        super().__init__(config)
        student_hybrid_conv_structured.install_conv_layers(self, config)


def inference_config(config: dict) -> dict:
    return student_multi_token.inference_config(config)


def inference_state(state: dict) -> dict:
    return student_multi_token.inference_state(state)


def build_model(config: dict) -> HybridConvMultiTokenLM:
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Multi-token parent source changed")
    return HybridConvMultiTokenLM(config)
