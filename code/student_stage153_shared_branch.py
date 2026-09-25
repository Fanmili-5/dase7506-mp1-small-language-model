"""Causal shared-trunk attention/conv and conv/attention branch R-Drop LM."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import student
import student_hybrid_conv_rdrop
from student_hybrid_conv_structured import GatedCausalConvBlock


PARENT_SHA = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
class SharedBranchLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        if (int(config["depth"]) != 8
                or tuple(config["conv_layers"]) != (2, 4, 6, 8)
                or tuple(config["branch_b_layers"]) != ("conv", "attention")
                or float(config["branch_mixture"]) != 0.5):
            raise ValueError("Unexpected fixed shared-branch architecture")
        super().__init__(config)
        self.branch_b = nn.ModuleList([
            GatedCausalConvBlock(config),
            student.Block(config),
        ])
        self.branch_b.apply(self._initialize)
        nn.init.normal_(self.branch_b[0].depthwise.weight, std=0.02)
        if self.branch_b[0].depthwise.bias is not None:
            nn.init.zeros_(self.branch_b[0].depthwise.bias)
        if bool(config.get("scaled_residual_init", False)):
            residual_std = 0.02 / math.sqrt(2 * len(self.blocks))
            nn.init.normal_(self.branch_b[0].proj.weight, std=residual_std)
            nn.init.normal_(self.branch_b[0].mlp.output.weight, std=residual_std)
            nn.init.normal_(self.branch_b[1].proj.weight, std=residual_std)
            nn.init.normal_(self.branch_b[1].mlp.output.weight, std=residual_std)
        self.branch_b_norm = student.make_norm(
            str(config.get("norm", "rmsnorm")).lower(), int(config["width"]),
            float(config.get("norm_eps", 1e-5)))
        self.branch_log_weight = math.log(0.5)

    def branch_features(self, ids: torch.Tensor, collect_aux: bool = False):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected independent [batch,time<=256] inputs")
        x = self.input_embeddings(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.embedding_dropout, self.training)
        auxiliary_logits = []
        for layer, block in enumerate(self.blocks[:6], 1):
            x = block(x)
            if collect_aux and layer in self.deep_supervision_layers:
                index = self.deep_supervision_layers.index(layer)
                auxiliary_logits.append(self.head(self.auxiliary_norms[index](x)))
        a, b = x, x
        for block in self.blocks[6:]:
            a = block(a)
        for block in self.branch_b:
            b = block(b)
        return self.norm(a), self.branch_b_norm(b), auxiliary_logits

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        a, b, _ = self.branch_features(ids)
        return torch.cat((a, b), dim=-1)

    def mixture_log_probs(self, a: torch.Tensor, b: torch.Tensor,
                          ids: torch.Tensor) -> torch.Tensor:
        a_logp = self._prefix_log_probs(a.float(), ids)
        b_logp = self._prefix_log_probs(b.float(), ids)
        return torch.logaddexp(a_logp, b_logp) + self.branch_log_weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        a, b, _ = self.branch_features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self.mixture_log_probs(a, b, ids)

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        return self(ids)

    def single_training_pass(self, ids, targets, future_targets):
        if not self.training or ids.shape != targets.shape or ids.ndim != 2:
            raise ValueError("Expected train mode and matched [batch,time] inputs")
        if tuple(future_targets.shape) != (
                len(self.future_prediction_offsets), *ids.shape):
            raise ValueError("Future targets have unexpected shape")
        a, b, auxiliary_logits = self.branch_features(ids, collect_aux=True)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            primary_logp = self.mixture_log_probs(a, b, ids)
            primary = F.nll_loss(primary_logp.flatten(0, 1), targets.flatten())
            deep = torch.stack([
                F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
                for logits in auxiliary_logits
            ]).mean()
            future_losses = []
            for hidden in (a.float(), b.float()):
                for index, (norm, projection) in enumerate(
                        zip(self.future_norms, self.future_projections)):
                    logits = self.head(projection(norm(hidden)))
                    future_losses.append(F.cross_entropy(
                        logits.flatten(0, 1), future_targets[index].flatten()))
            future = torch.stack(future_losses).mean()
            total = (primary + self.deep_supervision_weight * deep
                     + self.future_prediction_weight * future)
        return total, primary_logp, {
            "primary": primary.detach(), "deep": deep.detach(),
            "future": future.detach(),
        }


def build_model(config: dict) -> SharedBranchLM:
    actual = hashlib.sha256(Path(student_hybrid_conv_rdrop.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA:
        raise ValueError("Hybrid R-Drop parent source changed")
    return SharedBranchLM(config)
