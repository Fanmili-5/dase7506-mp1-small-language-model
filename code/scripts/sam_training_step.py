"""One efficient, first-order SAM update; only used during training."""
from __future__ import annotations

import math
from typing import Callable

import torch


def sam_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ascent_closure: Callable[[], tuple[torch.Tensor, dict]],
    descent_closure: Callable[[], tuple[torch.Tensor, dict]],
    *,
    rho: float,
) -> dict:
    """Perturb from the microbatch gradient, then update from full-batch loss.

    Both closures must use only the supplied training batch. The descent
    gradient is clipped at norm 1.0 after all parameter perturbations are
    restored, exactly as in the Stage54 training loop.
    """
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    ascent_loss, _ = ascent_closure()
    if not bool(torch.isfinite(ascent_loss)):
        raise FloatingPointError("Non-finite SAM ascent loss")
    ascent_loss.backward()
    active = [(parameter, parameter.grad.detach()) for parameter in parameters
              if parameter.grad is not None]
    if not active:
        raise ValueError("SAM ascent produced no gradients")
    ascent_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient.float()) for _, gradient in active
    ]))
    if not bool(torch.isfinite(ascent_norm)) or not bool(ascent_norm > 0):
        raise FloatingPointError("Invalid SAM ascent gradient norm")
    scale = rho / ascent_norm
    perturbations = [
        (parameter, parameter.detach().clone(), gradient * scale)
        for parameter, gradient in active
    ]
    with torch.no_grad():
        for parameter, _, perturbation in perturbations:
            parameter.add_(perturbation)
    try:
        optimizer.zero_grad(set_to_none=True)
        descent_loss, parts = descent_closure()
        if not bool(torch.isfinite(descent_loss)):
            raise FloatingPointError("Non-finite SAM descent loss")
        descent_loss.backward()
    finally:
        with torch.no_grad():
            for parameter, original, _ in perturbations:
                parameter.copy_(original)
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Non-finite SAM descent gradient norm")
    optimizer.step()
    return {
        "loss": descent_loss.detach(),
        "parts": parts,
        "ascent_loss": ascent_loss.detach(),
        "ascent_gradient_norm": float(ascent_norm),
        "descent_gradient_norm": float(gradient_norm),
    }
