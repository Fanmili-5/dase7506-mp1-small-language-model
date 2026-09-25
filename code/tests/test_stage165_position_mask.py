"""Stage165 training-only mask and unchanged-inference contract."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop as control
import student_position_mask_rdrop as masked

ROOT = Path(__file__).resolve().parents[1]


class PositionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs/stage165_position_mask_rdrop.json").read_text(encoding="utf-8")
        )

    def test_initial_weights_and_eval_are_exact_control(self) -> None:
        control_config = dict(self.config)
        control_config.pop(masked.MASK_KEY)
        torch.manual_seed(17)
        baseline = control.build_model(control_config).eval()
        torch.manual_seed(17)
        candidate = masked.build_model(self.config).eval()
        self.assertEqual(baseline.state_dict().keys(), candidate.state_dict().keys())
        for key in baseline.state_dict():
            self.assertTrue(torch.equal(baseline.state_dict()[key], candidate.state_dict()[key]), key)
        self.assertEqual(
            masked.inference_config(self.config), control.inference_config(control_config)
        )
        ids = torch.randint(0, 2048, (2, 14))
        with torch.no_grad():
            predicted = candidate.predict_log_probs(ids)
            self.assertTrue(torch.equal(baseline.predict_log_probs(ids), predicted))
            self.assertLess(float(predicted.logsumexp(-1).abs().max()), 1e-5)
            altered = ids.clone()
            altered[0, 9:] = torch.randint(0, 2048, (5,))
            changed = candidate.predict_log_probs(altered)
            solo = candidate.predict_log_probs(ids[:1])
        self.assertLess(float((predicted[0, :9] - changed[0, :9]).abs().max()), 3e-5)
        self.assertLess(float((predicted[1] - changed[1]).abs().max()), 3e-5)
        self.assertLess(float((predicted[:1] - solo).abs().max()), 3e-5)

    def test_training_mask_is_whole_position_and_stochastic(self) -> None:
        config = dict(self.config, embedding_row_dropout=0.0, dropout=0.0)
        torch.manual_seed(17)
        model = masked.build_model(config).train()
        ids = torch.randint(0, 2048, (2, 128))
        clean = model.token(ids)
        torch.manual_seed(23)
        first = model.input_embeddings(ids)
        second = model.input_embeddings(ids)
        dropped = first.abs().sum(-1) == 0
        self.assertGreater(int(dropped.sum()), 0)
        self.assertLess(int(dropped.sum()), ids.numel())
        self.assertTrue(torch.equal(first[dropped], torch.zeros_like(first[dropped])))
        self.assertTrue(torch.allclose(first[~dropped] * .95, clean[~dropped], atol=1e-7))
        self.assertFalse(torch.equal(first, second))

    def test_rdrop_has_finite_gradients(self) -> None:
        torch.manual_seed(17)
        model = masked.build_model(self.config).train()
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


if __name__ == "__main__":
    unittest.main()
