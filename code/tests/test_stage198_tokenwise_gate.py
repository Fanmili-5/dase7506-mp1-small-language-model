"""Synthetic guards for the non-deployable Stage198 token-wise gate diagnostic."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_stage198_tokenwise_gate import (
    VOCAB, fit_one_step, score_gradient_rows, shifted_weight,
)


class TestTokenwiseGateDiagnostic(unittest.TestCase):
    def setUp(self):
        self.neural = torch.tensor([[.55, .30, .15]], dtype=torch.float64)
        self.count = torch.tensor([[.10, .50, .40]], dtype=torch.float64)
        self.weight = torch.tensor([.20], dtype=torch.float64)
        self.target = torch.tensor([1], dtype=torch.long)
        self.mixed = ((1 - self.weight[:, None]) * self.neural
                      + self.weight[:, None] * self.count)

    def logp(self, offsets):
        weight = shifted_weight(self.weight, offsets)
        raw = (1 - weight) * self.neural + weight * self.count
        normalized = raw / raw.sum(1, keepdim=True)
        return normalized.gather(1, self.target[:, None]).log().sum()

    def test_zero_offset_reproduces_normalized_baseline(self):
        zero = torch.zeros(3, dtype=torch.float64)
        self.assertTrue(torch.allclose(shifted_weight(self.weight, zero),
                                       self.weight[:, None].expand(-1, 3), atol=1e-12))
        self.assertAlmostEqual(float(self.logp(zero)),
                               float(self.mixed[:, 1].log().sum()), places=12)

    def test_exact_score_gradient_matches_finite_difference(self):
        analytical = score_gradient_rows(
            self.neural, self.count, self.weight, self.mixed, self.target)[0]
        epsilon = 1e-5
        for token in range(3):
            plus = torch.zeros(3, dtype=torch.float64)
            minus = torch.zeros(3, dtype=torch.float64)
            plus[token] = epsilon
            minus[token] = -epsilon
            numerical = (self.logp(plus) - self.logp(minus)) / (2 * epsilon)
            self.assertAlmostEqual(float(analytical[token]), float(numerical), places=6)

    def test_regularized_step_is_bounded(self):
        gradient = torch.zeros(VOCAB, dtype=torch.float64)
        fisher = torch.zeros(VOCAB, dtype=torch.float64)
        gradient[0] = 100
        gradient[1] = -100
        fitted = fit_one_step(gradient, fisher)
        self.assertEqual(fitted.shape, (VOCAB,))
        self.assertEqual(float(fitted[0]), 2.0)
        self.assertEqual(float(fitted[1]), -2.0)
        self.assertEqual(float(fitted[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
