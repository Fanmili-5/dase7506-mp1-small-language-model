"""Structural checks for every student experiment configuration."""
import json
import unittest
from pathlib import Path

import torch

from model import build_model as build_baseline
from student import build_model as build_student


ROOT = Path(__file__).resolve().parents[1]


class StudentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config_paths = sorted((ROOT / "configs").glob("student_*.json"))

    def test_all_configs_build_and_predict(self):
        self.assertTrue(self.config_paths)
        for path in self.config_paths:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text())
                model = build_student(config).eval()
                ids = torch.randint(0, 2048, (2, 17))
                with torch.no_grad():
                    logp = model.predict_log_probs(ids)
                self.assertEqual(tuple(logp.shape), (2, 17, 2048))
                torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, 17), atol=1e-6, rtol=1e-6)

    def test_control_matches_baseline_parameter_count(self):
        config = json.loads((ROOT / "configs/student_control.json").read_text())
        baseline_config = {key: config[key] for key in ("vocab", "width", "heads", "depth", "context")}
        baseline = build_baseline(baseline_config)
        control = build_student(config)
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        control_parameters = sum(parameter.numel() for parameter in control.parameters())
        self.assertEqual(baseline_parameters, 1_088_256)
        self.assertEqual(control_parameters, baseline_parameters)

    def test_scaled_config_parameter_count(self):
        expected = {
            "student_scaled.json": 3_049_920,
            "student_w192_d8.json": 3_935_424,
            "student_w224_d6.json": 4_072_992,
            "student_w256_d6.json": 5_247_744,
            "student_w256_d6_drop005.json": 5_247_744,
            "student_w256_d6_drop010.json": 5_247_744,
            "student_w256_d6_drop015.json": 5_247_744,
            "student_w256_d6_drop020.json": 5_247_744,
        }
        for name, parameters in expected.items():
            with self.subTest(config=name):
                config = json.loads((ROOT / "configs" / name).read_text())
                model = build_student(config)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameters)

    def test_all_configs_are_causal(self):
        for path in self.config_paths:
            with self.subTest(config=path.name):
                model = build_student(json.loads(path.read_text())).eval()
                ids = torch.randint(0, 2048, (2, 31))
                changed = ids.clone()
                changed[:, 13:] = (changed[:, 13:] + 1) % 2048
                with torch.no_grad():
                    original = model.predict_log_probs(ids)
                    perturbed = model.predict_log_probs(changed)
                torch.testing.assert_close(original[:, :13], perturbed[:, :13], atol=1e-6, rtol=1e-6)

    def test_all_configs_have_independent_examples_and_windows(self):
        for path in self.config_paths:
            with self.subTest(config=path.name):
                model = build_student(json.loads(path.read_text())).eval()
                ids = torch.randint(0, 2048, (2, 31))
                with torch.no_grad():
                    original = model.predict_log_probs(ids)
                    first_alone = model.predict_log_probs(ids[:1])
                    model.predict_log_probs(torch.randint(0, 2048, (3, 19)))
                    repeated = model.predict_log_probs(ids)
                torch.testing.assert_close(original[:1], first_alone, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(original, repeated, atol=0, rtol=0)

    def test_control_initialization_and_logits_match_baseline(self):
        config = json.loads((ROOT / "configs/student_control.json").read_text())
        baseline_config = {key: config[key] for key in ("vocab", "width", "heads", "depth", "context")}
        torch.manual_seed(123)
        baseline = build_baseline(baseline_config).eval()
        torch.manual_seed(123)
        control = build_student(config).eval()
        ids = torch.randint(0, 2048, (2, 19))
        with torch.no_grad():
            expected = baseline(ids)
            actual = control(ids)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_rope_ablation_changes_only_position_fields(self):
        control = json.loads((ROOT / "configs/student_control.json").read_text())
        rope = json.loads((ROOT / "configs/student_rope.json").read_text())
        differing = {key for key in control.keys() | rope.keys() if control.get(key) != rope.get(key)}
        self.assertEqual(differing, {"position", "rope_base"})

    def test_stage2_ladder_changes_only_declared_fields(self):
        names = ["rope", "rope_swiglu", "rope_swiglu_rms",
                 "rope_swiglu_rms_nobias", "modern"]
        expected = [{"activation", "mlp_ratio"}, {"norm"}, {"bias"},
                    {"scaled_residual_init"}]
        configs = []
        for name in names:
            config = json.loads((ROOT / f"configs/student_{name}.json").read_text())
            config.setdefault("norm_eps", 1e-5)
            configs.append(config)
        for left, right, fields in zip(configs, configs[1:], expected):
            differing = {key for key in left.keys() | right.keys()
                         if left.get(key) != right.get(key)}
            self.assertEqual(differing, fields)


if __name__ == "__main__":
    unittest.main()
