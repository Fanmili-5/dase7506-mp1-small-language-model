"""Stage165 training-only whole-position input-embedding masking."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

import student_hybrid_conv_rdrop

PARENT_SHA256 = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
MASK_KEY = "position_embedding_mask"


class PositionMaskRDropLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        probability = float(config[MASK_KEY])
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError("position_embedding_mask must be in [0,1)")
        self.position_embedding_mask = probability

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().input_embeddings(ids)
        if not self.training or self.position_embedding_mask == 0.0:
            return embeddings
        keep = 1.0 - self.position_embedding_mask
        row_mask = (torch.rand((*ids.shape, 1), device=ids.device) < keep)
        return embeddings * row_mask.to(embeddings.dtype) / keep


def inference_config(config: dict) -> dict:
    result = student_hybrid_conv_rdrop.inference_config(config)
    result.pop(MASK_KEY, None)
    return result


def inference_state(state: dict) -> dict:
    return student_hybrid_conv_rdrop.inference_state(state)


def build_model(config: dict) -> PositionMaskRDropLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Pinned hybrid training parent changed")
    return PositionMaskRDropLM(config)
