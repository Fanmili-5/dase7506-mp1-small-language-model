"""Small analytic tests for the train-only Stage200 target margin."""
import math
import unittest

import torch
from torch.nn import functional as F

from scripts.train_stage200_target_margin_pilot import target_margin_nll


class TargetMarginTest(unittest.TestCase):
    def setUp(self):
        logits = torch.tensor([[[1.1, -0.2, 0.5], [0.1, 0.8, -1.4]]])
        self.logp = F.log_softmax(logits, dim=-1)
        self.targets = torch.tensor([[0, 2]])

    def test_zero_margin_is_ordinary_nll(self):
        actual = target_margin_nll(
            self.logp, self.targets, torch.zeros_like(self.targets, dtype=torch.float))
        expected = F.nll_loss(self.logp.flatten(0, 1), self.targets.flatten())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_matches_direct_target_logit_reduction(self):
        margin = torch.tensor([[0.2, 0.7]])
        actual = target_margin_nll(self.logp, self.targets, margin)
        modified = self.logp.clone().scatter_add(
            -1, self.targets.unsqueeze(-1), -margin.unsqueeze(-1))
        expected = F.cross_entropy(modified.flatten(0, 1), self.targets.flatten())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_finite_gradients_and_invalid_shapes(self):
        logp = self.logp.detach().requires_grad_()
        loss = target_margin_nll(logp, self.targets, torch.full((1, 2), 0.3))
        loss.backward()
        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(torch.isfinite(logp.grad).all())
        with self.assertRaises(ValueError):
            target_margin_nll(logp, self.targets, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
