"""Single-GPU Muon pilot for hidden matrices only.

Adapted from Keller Jordan et al., https://github.com/KellerJordan/Muon/blob/master/muon.py
(MIT license). The original Newton--Schulz coefficients, Nesterov momentum, and
rectangular-matrix update scale are retained. Embeddings, norms, biases, and
output-related parameters remain with PyTorch AdamW in the experiment script.
"""

import torch


def newton_schulz_zeroth_power(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if gradient.ndim != 2:
        raise ValueError("Muon pilot accepts only 2D hidden weights")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.to(torch.bfloat16)
    transposed = gradient.shape[0] > gradient.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    return x.T if transposed else x


class HiddenMatrixMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.01):
        params = list(params)
        if not params or any(param.ndim != 2 for param in params):
            raise ValueError("Muon parameters must be nonempty 2D hidden matrices")
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon pilot does not support closures")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                gradient = param.grad
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Non-finite Muon gradient")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum = state["momentum_buffer"]
                momentum.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(momentum, group["momentum"])
                update = newton_schulz_zeroth_power(update)
                update = update * max(1.0, param.shape[0] / param.shape[1]) ** 0.5
                if not torch.isfinite(update).all():
                    raise FloatingPointError("Non-finite Muon update")
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update.to(param.dtype), alpha=-group["lr"])


def partition_hidden_matrices(model):
    """Exactly partition named parameters; no tied embedding can enter Muon."""
    muon, adam = [], []
    names = {"muon": [], "adamw": []}
    for name, param in model.named_parameters():
        if name.startswith("blocks.") and name.endswith(".weight") and param.ndim == 2:
            muon.append(param)
            names["muon"].append(name)
        else:
            adam.append(param)
            names["adamw"].append(name)
    all_ids = [id(param) for param in model.parameters()]
    selected_ids = [id(param) for param in muon + adam]
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != set(all_ids):
        raise ValueError("Optimizer groups are not disjoint and exhaustive")
    return muon, adam, names
