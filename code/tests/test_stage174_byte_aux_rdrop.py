"""Stage174 train-only auxiliary supervision and unchanged inference contract."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop as control
import student_hybrid_conv_structured as deployed
import student_stage174_byte_aux_rdrop as candidate


ROOT = Path(__file__).resolve().parents[1]


class ByteAuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/stage174_byte_aux_rdrop.json").read_text())
        cls.control_config = dict(cls.config)
        cls.control_config.pop(candidate.AUX_WEIGHT_KEY)

    def test_initial_shared_weights_and_eval_output_are_exact(self) -> None:
        torch.manual_seed(17)
        baseline = control.build_model(self.control_config).eval()
        torch.manual_seed(17)
        model = candidate.build_model(self.config).eval()
        for name, weight in baseline.state_dict().items():
            self.assertTrue(torch.equal(weight, model.state_dict()[name]), name)
        self.assertEqual(candidate.inference_config(self.config),
                         control.inference_config(self.control_config))
        self.assertEqual(set(candidate.inference_state(model.state_dict())),
                         set(control.inference_state(baseline.state_dict())))
        ids = (torch.arange(28).reshape(2, 14) * 23 + 17) % 2048
        with torch.inference_mode():
            expected = baseline.predict_log_probs(ids)
            actual = model.predict_log_probs(ids)
        self.assertTrue(torch.equal(actual, expected))
        self.assertLess(float(actual.logsumexp(-1).abs().max()), 1e-5)
        export = deployed.build_model(candidate.inference_config(self.config)).eval()
        export.load_state_dict(candidate.inference_state(model.state_dict()), strict=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(export.predict_log_probs(ids), expected))

    def test_causal_and_independent_rows(self) -> None:
        torch.manual_seed(17)
        model = candidate.build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 13))
        altered = ids.clone()
        altered[0, 8:] = torch.randint(0, 2048, (5,))
        with torch.inference_mode():
            original = model.predict_log_probs(ids)
            changed = model.predict_log_probs(altered)
            alone = model.predict_log_probs(ids[:1])
        self.assertLess(float((original[0, :8] - changed[0, :8]).abs().max()), 3e-5)
        self.assertLess(float((original[1] - changed[1]).abs().max()), 3e-5)
        self.assertLess(float((original[:1] - alone).abs().max()), 3e-5)

    def test_byte_targets_and_training_gradients(self) -> None:
        torch.manual_seed(17)
        model = candidate.build_model(self.config).train()
        self.assertEqual(model.first_byte.shape, (2048,))
        self.assertEqual(model.last_byte.shape, (2048,))
        self.assertGreaterEqual(int(model.first_byte.min()), 0)
        self.assertLessEqual(int(model.last_byte.max()), 255)
        self.assertNotIn("first_byte", model.state_dict())
        self.assertNotIn("last_byte", model.state_dict())
        ids = torch.randint(0, 2048, (2, 8))
        targets = torch.randint(0, 2048, (2, 8))
        future = torch.randint(0, 2048, (2, 2, 8))
        total, parts = model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(parts["byte_aux"]))
        self.assertGreater(float(parts["byte_aux"]), 0)
        total.backward()
        for name in ("aux_first.weight", "aux_last.weight", "token.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)


if __name__ == "__main__":
    unittest.main()
