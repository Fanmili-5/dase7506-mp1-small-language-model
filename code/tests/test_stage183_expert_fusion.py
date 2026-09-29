"""Synthetic distributions only; no course validation or test text."""
from __future__ import annotations

import unittest

import torch

from scripts.diagnose_stage183_expert_fusion import BETAS, geometric_expert


class ExpertFusionTests(unittest.TestCase):
    def test_geometric_normalizes_and_beta_zero_is_original(self) -> None:
        neural = torch.full((1, 2, 2048), 1 / 2048)
        count = neural.clone()
        neural[0, 0, :2] = torch.tensor([0.4, 0.1])
        neural[0, 0, 2:] = 0.5 / 2046
        count[0, 0, :2] = torch.tensor([0.1, 0.4])
        count[0, 0, 2:] = 0.5 / 2046
        weight = torch.tensor([[0.2, 0.5]])
        arithmetic = (1 - weight).unsqueeze(-1) * neural + weight.unsqueeze(-1) * count
        geometric = geometric_expert(neural, count, weight)
        self.assertTrue(torch.allclose(geometric.sum(-1), torch.ones((1, 2)), atol=1e-6))
        self.assertTrue(torch.equal((1 - BETAS[0]) * arithmetic + BETAS[0] * geometric,
                                    arithmetic))
        self.assertGreater((geometric - arithmetic).abs().max().item(), 1e-3)

    def test_invalid_weight_or_expert_is_rejected(self) -> None:
        neural = torch.full((1, 1, 2048), 1 / 2048)
        count = neural.clone()
        with self.assertRaises(ValueError):
            geometric_expert(neural, count, torch.tensor([[0.0]]))
        count[0, 0, 0] = -0.1
        with self.assertRaises(ValueError):
            geometric_expert(neural, count, torch.tensor([[0.1]]))


if __name__ == "__main__":
    unittest.main()
