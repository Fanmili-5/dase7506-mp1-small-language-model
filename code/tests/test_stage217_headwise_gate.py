"""Zero-start, causality and gradient checks for Stage217."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import student_hybrid_conv_rdrop
import student_stage217_headwise_gate


class Stage217HeadwiseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage217_headwise_gate_rdrop.json").read_text())
        cls.control = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())

    def test_config_only_adds_gate(self):
        changed = {key for key in self.config.keys() | self.control.keys()
                   if self.config.get(key) != self.control.get(key)}
        self.assertEqual(changed, {"attention_gate_kind"})
        self.assertEqual(self.config["attention_gate_kind"], "headwise_post_sdpa_zero_start")

    def test_zero_start_normalization_and_causality(self):
        torch.manual_seed(217017)
        control = student_hybrid_conv_rdrop.build_model(self.control).eval()
        torch.manual_seed(217017)
        candidate = student_stage217_headwise_gate.build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 14))
        with torch.no_grad():
            reference = control.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
            self.assertLessEqual(float((actual - reference).abs().max()), 1e-5)
            self.assertLessEqual(float(actual.logsumexp(-1).abs().max()), 1e-5)
            modified = ids.clone()
            modified[:, 9:] = torch.randint(0, 2048, (2, 5))
            prefix = candidate.predict_log_probs(modified)
            self.assertLessEqual(float((prefix[:, :9] - actual[:, :9]).abs().max()), 3e-5)
            self.assertLessEqual(float((candidate.predict_log_probs(ids[:1])[0]
                                        - actual[0]).abs().max()), 1e-5)

    def test_finite_gate_gradients(self):
        torch.manual_seed(217017)
        model = student_stage217_headwise_gate.build_model(self.config).train()
        ids = torch.randint(0, 2048, (1, 14))
        target = torch.randint(0, 2048, (1, 14))
        loss = torch.nn.functional.nll_loss(model(ids).flatten(0, 1), target.flatten())
        loss.backward()
        for block in model.blocks:
            if isinstance(block, student_stage217_headwise_gate.HeadwiseGatedAttentionBlock):
                self.assertTrue(torch.isfinite(block.head_gate.weight.grad).all())
                self.assertTrue(torch.isfinite(block.head_gate.bias.grad).all())


if __name__ == "__main__":
    unittest.main()
