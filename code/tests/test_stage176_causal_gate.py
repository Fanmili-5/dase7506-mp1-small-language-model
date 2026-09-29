"""Synthetic guards for the non-deployable Stage176 gate diagnostic."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_stage176_causal_gate import ResidualGate, mixture_nll


class TestCausalGateDiagnostic(unittest.TestCase):
    def test_zero_start_exact_gate(self):
        torch.manual_seed(1)
        gate = ResidualGate(294)
        inputs = torch.randn(8, 294)
        old = torch.tensor([.01, .05, .1, .15, .25, .4, .7, .9])
        self.assertTrue(torch.allclose(gate(inputs, old), old, atol=1e-7))

    def test_normalized_target_mixture_loss(self):
        weight = torch.tensor([.25, .75])
        neural = torch.tensor([.3, .1])
        count = torch.tensor([.1, .4])
        expected = -((1 - weight) * neural + weight * count).double().log()
        self.assertTrue(torch.allclose(mixture_nll(weight, neural, count),
                                       expected, atol=1e-7))

    def test_rejects_invalid_target_probability(self):
        with self.assertRaises(ValueError):
            mixture_nll(torch.tensor([.5]), torch.tensor([0.]),
                        torch.tensor([.2]))


if __name__ == "__main__":
    unittest.main()
