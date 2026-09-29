"""Inference view of the Stage164 multiscale causal-convolution backbone."""
from __future__ import annotations

import hashlib
from pathlib import Path

import student_hybrid_conv_structured

PARENT_SHA256 = "81e61cc36bc6aec3711de27c9860c7ac6b80238b8b5481c5bfa7710a04d9c34e"


def install_dilations(model, config: dict) -> None:
    layers = tuple(int(layer) for layer in config["conv_layers"])
    dilations = tuple(int(value) for value in config["conv_dilations"])
    if len(dilations) != len(layers) or any(value < 1 for value in dilations):
        raise ValueError("conv_dilations must have one positive value per conv layer")
    for layer, dilation in zip(layers, dilations):
        block = model.blocks[layer - 1]
        if not isinstance(block, student_hybrid_conv_structured.GatedCausalConvBlock):
            raise TypeError(f"layer {layer} is not a causal convolution block")
        learned_kernel = block.depthwise.kernel_size[0]
        # GatedCausalConvBlock.forward uses `kernel` only to left-pad. Its
        # Conv1d keeps the same seven learned taps and parameter names.
        block.depthwise.dilation = (dilation,)
        block.kernel = (learned_kernel - 1) * dilation + 1
    model.conv_dilations = dilations


class DilatedHybridStructuredLM(student_hybrid_conv_structured.HybridConvStructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_dilations(self, config)


def build_model(config: dict) -> DilatedHybridStructuredLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Pinned hybrid inference parent changed")
    return DilatedHybridStructuredLM(config)
