"""Stage211 exact-start and train-only auxiliary-head invariants."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_hybrid_conv_structured
from scripts.preflight_stage211_long_horizon import build_matched_candidate


ROOT = Path(__file__).resolve().parents[1]


class Stage211LongHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/stage211_long_horizon_rdrop.json").read_text()
        )
        cls.control_config = json.loads(
            (ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text()
        )

    def test_only_offsets_change_and_initial_inference_is_exact(self):
        changed = {key for key in self.config.keys() | self.control_config.keys()
                   if self.config.get(key) != self.control_config.get(key)}
        self.assertEqual(changed, {"future_prediction_offsets"})
        self.assertEqual(self.config["future_prediction_offsets"],
                         [2, 3, 8, 16, 32, 64])
        torch.manual_seed(17)
        control, candidate, _ = build_matched_candidate(
            self.config, self.control_config, torch.device("cpu"))
        control.eval(); candidate.eval()
        ids = torch.randint(2048, (2, 16))
        with torch.inference_mode():
            expected = control.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual.logsumexp(-1),
                                       torch.zeros_like(actual[..., 0]),
                                       atol=2e-6, rtol=0)
            changed_ids = ids.clone()
            changed_ids[0, -1] = (changed_ids[0, -1] + 1) % 2048
            torch.testing.assert_close(
                candidate.predict_log_probs(changed_ids)[0, :-1],
                actual[0, :-1], atol=2e-6, rtol=0)
            torch.testing.assert_close(
                candidate.predict_log_probs(ids[:1])[0],
                actual[0], atol=2e-6, rtol=0)

    def test_inference_export_omits_all_six_auxiliary_heads(self):
        torch.manual_seed(17)
        control, candidate, _ = build_matched_candidate(
            self.config, self.control_config, torch.device("cpu"))
        self.assertEqual(student_hybrid_conv_rdrop.inference_config(self.config),
                         student_hybrid_conv_rdrop.inference_config(
                             self.control_config))
        state = student_hybrid_conv_rdrop.inference_state(candidate.state_dict())
        control_state = student_hybrid_conv_rdrop.inference_state(
            control.state_dict())
        self.assertEqual(state.keys(), control_state.keys())
        self.assertFalse(any(key.startswith(("future_norms.",
                                             "future_projections."))
                             for key in state))
        for key in state:
            torch.testing.assert_close(state[key], control_state[key],
                                       atol=0, rtol=0)
        deployed = student_hybrid_conv_structured.build_model(
            student_hybrid_conv_rdrop.inference_config(self.config)).eval()
        deployed.load_state_dict(state, strict=True)
        candidate.eval()
        ids = torch.randint(2048, (1, 16))
        with torch.inference_mode():
            torch.testing.assert_close(deployed.predict_log_probs(ids),
                                       candidate.predict_log_probs(ids),
                                       atol=0, rtol=0)

    def test_six_training_heads_have_finite_gradients(self):
        torch.manual_seed(17)
        _, candidate, _ = build_matched_candidate(
            self.config, self.control_config, torch.device("cpu"))
        candidate.train()
        ids = torch.randint(2048, (2, 16))
        targets = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (6, *ids.shape))
        loss, _ = candidate.rdrop_training_loss(ids, targets, future)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(len(candidate.future_projections), 6)
        for head in candidate.future_projections:
            grad = head.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
