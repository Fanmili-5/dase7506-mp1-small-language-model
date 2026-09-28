"""Tie the last logical attention/conv pair of a ten-block hybrid LM."""
from __future__ import annotations

from torch import nn

import student
from student_hybrid_conv_structured import GatedCausalConvBlock


def tie_tail_pair(model: nn.Module, config: dict) -> None:
    if (int(config.get("depth", 0)) != 10
            or tuple(config.get("conv_layers", ())) != (2, 4, 6, 8, 10)
            or tuple(config.get("shared_tail_pair", ())) != (7, 8)
            or len(model.blocks) != 10):
        raise ValueError("Stage221 fixes a ten-block alternating hybrid")
    if (not isinstance(model.blocks[6], student.Block)
            or not isinstance(model.blocks[7], GatedCausalConvBlock)
            or not isinstance(model.blocks[8], student.Block)
            or not isinstance(model.blocks[9], GatedCausalConvBlock)):
        raise ValueError("Stage221 expected the final attention/conv pairs")
    model.blocks[8] = model.blocks[6]
    model.blocks[9] = model.blocks[7]
    if model.blocks[8] is not model.blocks[6] or model.blocks[9] is not model.blocks[7]:
        raise AssertionError("Tail modules were not tied")
