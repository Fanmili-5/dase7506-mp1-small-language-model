"""Training-only stochastic depth for the fixed Stage54 matched pilot.

The hooks have no parameters or buffers. In evaluation mode they return the
unmodified block output, so the ordinary Stage54 inference model/state remain
exactly the same architecture.
"""
from __future__ import annotations

import torch


def attach_drop_path(model: torch.nn.Module, rate: float) -> None:
    if not 0.0 <= rate < 1.0:
        raise ValueError("drop-path rate must be in [0, 1)")
    if getattr(model, "_drop_path_attached", False):
        raise ValueError("drop path already attached")

    def hook(module: torch.nn.Module, inputs: tuple, output: torch.Tensor):
        if not module.training or rate == 0.0:
            return output
        residual = inputs[0]
        if not isinstance(output, torch.Tensor) or output.shape != residual.shape:
            raise ValueError("drop path requires a shape-preserving tensor block")
        keep = (torch.rand((output.shape[0], 1, 1), device=output.device)
                >= rate).to(output.dtype) / (1.0 - rate)
        return residual + (output - residual) * keep

    for block in model.blocks:
        block.register_forward_hook(hook)
    model._drop_path_attached = True
