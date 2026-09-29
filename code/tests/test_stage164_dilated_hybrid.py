"""Stage164 structural checks before any validation-quality pilot."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

import student_dilated_hybrid_rdrop as dilated
import student_hybrid_conv_rdrop as control

ROOT = Path(__file__).resolve().parents[1]


class DilatedHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs/stage164_dilated_hybrid_rdrop.json").read_text(encoding="utf-8")
        )

    def test_same_initial_weights_and_all_one_equivalence(self) -> None:
        baseline_config = dict(self.config)
        baseline_config.pop("conv_dilations")
        torch.manual_seed(17)
        baseline = control.build_model(baseline_config).eval()
        torch.manual_seed(17)
        candidate = dilated.build_model(self.config).eval()
        self.assertEqual(baseline.state_dict().keys(), candidate.state_dict().keys())
        for key in baseline.state_dict():
            self.assertTrue(torch.equal(baseline.state_dict()[key], candidate.state_dict()[key]), key)
        self.assertEqual(candidate.conv_dilations, (1, 2, 4, 8))
        self.assertEqual([candidate.blocks[i].kernel for i in (1, 3, 5, 7)], [7, 13, 25, 49])

        all_one = dict(self.config, conv_dilations=[1, 1, 1, 1])
        torch.manual_seed(17)
        identical = dilated.build_model(all_one).eval()
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline(ids), identical(ids)))

    def test_causal_normalized_and_independent(self) -> None:
        torch.manual_seed(17)
        model = dilated.build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 14))
        altered = ids.clone()
        altered[0, 9:] = torch.randint(0, 2048, (5,))
        with torch.no_grad():
            original = model.predict_log_probs(ids)
            changed = model.predict_log_probs(altered)
            solo = model.predict_log_probs(ids[:1])
        self.assertEqual(tuple(original.shape), (2, 14, 2048))
        self.assertTrue(torch.isfinite(original).all())
        self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float((original[0, :9] - changed[0, :9]).abs().max()), 3e-5)
        self.assertLess(float((original[1] - changed[1]).abs().max()), 3e-5)
        self.assertLess(float((original[:1] - solo).abs().max()), 3e-5)

    def test_finite_training_gradients(self) -> None:
        torch.manual_seed(17)
        model = dilated.build_model(self.config).train()
        ids = torch.randint(0, 2048, (2, 8))
        targets = torch.randint(0, 2048, (2, 8))
        future = torch.randint(0, 2048, (2, 2, 8))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        self.assertTrue(all(
            model.blocks[index].depthwise.weight.grad is not None
            for index in (1, 3, 5, 7)
        ))


if __name__ == "__main__":
    unittest.main()
