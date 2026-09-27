"""Synthetic causal and zero-start contracts for the Stage207 lexical head."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

import student_stage207_prefix_residual as stage207


class PrefixResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(207017)
        torch.set_num_threads(4)
        cls.start = next(i for i, value in enumerate(stage207.STARTS)
                         if value is not None and len(value) == 2)
        cls.cont = next(i for i, value in enumerate(stage207.CONTINUATIONS)
                        if value is not None and len(value) == 1)
        cls.break_id = next(i for i in range(stage207.VOCAB)
                            if stage207.STARTS[i] is None
                            and stage207.CONTINUATIONS[i] is None)

    def test_prefix_resets_on_nonword_and_row_boundary(self):
        ids = torch.tensor([[self.start, self.cont, self.break_id, self.cont],
                            [self.cont, self.start, self.cont, self.break_id]])
        chars, active = stage207.prefix_features(ids)
        self.assertEqual(chars.shape, (2, 4, stage207.FEATURE_LETTERS))
        self.assertEqual(active.tolist(), [[1., 1., 0., 0.], [0., 1., 1., 0.]])
        self.assertEqual(chars[0, 0, -2:].tolist(),
                         [stage207._char_id(c) for c in stage207.STARTS[self.start]])

    def test_zero_start_normalization_causality_and_rows(self):
        model = stage207.PrefixConditionedResidual().eval()
        ids = torch.tensor([[self.start, self.cont, self.break_id, self.cont],
                            [self.cont, self.start, self.cont, self.break_id]])
        base = F.log_softmax(torch.randn(2, 4, 2048), dim=-1)
        hidden = torch.randn(2, 4, 288)
        with torch.inference_mode():
            got = model(base, hidden, ids)
            changed = ids.clone()
            changed[0, -1] = self.start
            changed_got = model(base, hidden, changed)
            row = model(base[:1], hidden[:1], ids[:1])
        self.assertLess(float((got - base).abs().max()), 2e-6)
        self.assertLess(float(got.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float((got[0, :-1] - changed_got[0, :-1]).abs().max()), 1e-6)
        self.assertLess(float((got[:1] - row).abs().max()), 1e-6)

    def test_residual_has_trainable_gradient(self):
        model = stage207.PrefixConditionedResidual().train()
        ids = torch.tensor([[self.start, self.cont]])
        base = F.log_softmax(torch.randn(1, 2, 2048), dim=-1)
        hidden = torch.randn(1, 2, 288)
        targets = torch.randint(0, 2048, (1, 2))
        loss = F.nll_loss(model(base, hidden, ids).flatten(0, 1), targets.flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.output.weight.grad)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
