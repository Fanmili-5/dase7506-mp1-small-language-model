"""Zero-start, causality and finite-gradient checks for Stage216."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import student_hybrid_conv_rdrop
import student_stage216_relative_bias


class Stage216RelativeBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage216_relative_bias_rdrop.json").read_text())
        cls.control = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())

    def test_config_and_buckets(self):
        changed = {key for key in self.config.keys() | self.control.keys()
                   if self.config.get(key) != self.control.get(key)}
        self.assertEqual(changed, {"relative_bias_buckets"})
        buckets = student_stage216_relative_bias.distance_buckets(256)
        self.assertEqual(tuple(buckets.shape), (256, 256))
        self.assertTrue(torch.equal(buckets.diag(), torch.zeros(256, dtype=torch.long)))
        self.assertEqual(int(buckets.max()), 31)
        self.assertEqual(int(buckets[15, 0]), 15)
        self.assertEqual(int(buckets[16, 0]), 16)
        self.assertEqual(int(buckets[255, 0]), 31)

    def test_zero_start_and_causality(self):
        torch.manual_seed(216017)
        control = student_hybrid_conv_rdrop.build_model(self.control).eval()
        torch.manual_seed(216017)
        candidate = student_stage216_relative_bias.build_model(self.config).eval()
        ids = torch.randint(0, 2048, (1, 12))
        with torch.no_grad():
            reference = control.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
            self.assertLessEqual(float((actual - reference).abs().max()), 1e-5)
            changed = ids.clone()
            changed[0, 8:] = torch.randint(0, 2048, (4,))
            earlier = candidate.predict_log_probs(changed)
            self.assertLessEqual(float((earlier[:, :8] - actual[:, :8]).abs().max()), 3e-5)
            self.assertLessEqual(float(actual.logsumexp(-1).abs().max()), 1e-5)

    def test_finite_gradient_on_bias(self):
        torch.manual_seed(216017)
        model = student_stage216_relative_bias.build_model(self.config).train()
        ids = torch.randint(0, 2048, (1, 12))
        target = torch.randint(0, 2048, (1, 12))
        loss = torch.nn.functional.nll_loss(model(ids).flatten(0, 1), target.flatten())
        loss.backward()
        for block in model.blocks:
            if isinstance(block, student_stage216_relative_bias.RelativeBiasAttentionBlock):
                self.assertIsNotNone(block.relative_bias.grad)
                self.assertTrue(torch.isfinite(block.relative_bias.grad).all())


if __name__ == "__main__":
    unittest.main()
