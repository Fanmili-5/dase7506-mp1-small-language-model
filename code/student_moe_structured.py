"""Top-1 routed SwiGLU capacity for the structured Transformer.

Routing is causal because every decision uses only the hidden state at the same
position.  Each token is evaluated by exactly one expert.  The implementation
does not drop tokens or carry state between independent evaluation windows.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_structured

PARENT_SHA = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class Top1SwiGLU(nn.Module):
    """Two or more SwiGLU experts with hard top-1 token routing."""

    def __init__(self, width: int, hidden: int, experts: int, bias: bool,
                 gate_scale: float):
        super().__init__()
        if experts < 2 or hidden < 1:
            raise ValueError("MoE requires at least two non-empty experts")
        if not math.isfinite(gate_scale) or gate_scale <= 0:
            raise ValueError("moe_gate_scale must be finite and positive")
        self.expert_count = experts
        self.gate_scale = gate_scale
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(
            [student.SwiGLU(width, hidden, bias) for _ in range(experts)]
        )
        self.router_balance_loss = None
        self.router_z_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat)
        probabilities = F.softmax(router_logits.float(), dim=-1)
        choices = probabilities.argmax(dim=-1)
        if self.training:
            # Large dense GEMMs are much faster than many indexed small GEMMs on
            # the training GPU. Only the chosen expert output contributes, so
            # routing semantics and gradients are unchanged. Evaluation remains
            # genuinely sparse because CPU time is part of the course budget.
            all_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
            rows = torch.arange(flat.shape[0], device=flat.device)
            selected = all_outputs[rows, choices]
            gate = probabilities[rows, choices]
            output = selected * (self.gate_scale * gate).to(selected.dtype)[:, None]
            output = output.to(flat.dtype)
            fractions = F.one_hot(choices, self.expert_count).float().mean(dim=0)
            mean_probabilities = probabilities.mean(dim=0)
            self.router_balance_loss = self.expert_count * (
                fractions.detach() * mean_probabilities
            ).sum()
            self.router_z_loss = router_logits.float().logsumexp(dim=-1).square().mean()
        else:
            output = torch.zeros_like(flat)
            for expert_index, expert in enumerate(self.experts):
                indices = torch.nonzero(choices == expert_index, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                selected = flat.index_select(0, indices)
                expert_output = expert(selected)
                gate = probabilities.index_select(0, indices)[:, expert_index]
                expert_output = expert_output * (
                    self.gate_scale * gate).to(expert_output.dtype)[:, None]
                output = output.index_copy(0, indices, expert_output.to(output.dtype))
            self.router_balance_loss = None
            self.router_z_loss = None
        return output.reshape(shape)


def install_moe_layers(model: nn.Module, config: dict) -> None:
    """Replace configured dense FFNs without changing attention or residual paths."""
    layers = tuple(int(value) for value in config.get("moe_layers", ()))
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("moe_layers must be unique and increasing")
    if layers[0] < 1 or layers[-1] > len(model.blocks):
        raise ValueError("moe_layers contains an invalid one-based layer index")
    width = int(config["width"])
    hidden = max(1, int(round(width * float(config["moe_mlp_ratio"]))))
    experts = int(config.get("moe_experts", 2))
    bias = bool(config.get("bias", True))
    gate_scale = float(config.get("moe_gate_scale", experts))
    depth = len(model.blocks)
    for layer in layers:
        routed = Top1SwiGLU(width, hidden, experts, bias, gate_scale)
        routed.apply(model._initialize)
        nn.init.normal_(routed.router.weight, std=0.01)
        if bool(config.get("scaled_residual_init", False)):
            residual_std = 0.02 / math.sqrt(2 * depth)
            for expert in routed.experts:
                nn.init.normal_(expert.output.weight, std=residual_std)
        model.blocks[layer - 1].mlp = routed
    model.moe_layers = layers


class MoEStructuredLM(student_structured.StructuredLM):
    def __init__(self, config: dict):
        super().__init__(config)
        install_moe_layers(self, config)


def build_model(config: dict) -> MoEStructuredLM:
    actual = hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Structured parent source changed")
    return MoEStructuredLM(config)
