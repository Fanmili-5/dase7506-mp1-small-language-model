"""Architecture-specific contract, causality, gradient and reconstruction tests."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from common import make_model
from student_structured import build_model

ROOT = Path(__file__).resolve().parents[1]


class StructuredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, kind):
        return dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                    position="rope", norm="rmsnorm", activation="swiglu",
                    mlp_ratio=2.6666667, bias=False, dropout=0.1,
                    output_kind=kind, copy_dim=8, mos_components=2)

    def test_actual_configs_counts_and_full_context(self):
        cases = [("architecture_a_w320_d4", "student", 5572160),
                 ("architecture_b_copy64", "student_structured", 5280769),
                 ("architecture_c_mos2", "student_structured", 5379842)]
        for name, implementation, count in cases:
            with self.subTest(name=name):
                config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
                model, _ = make_model(implementation, config, torch.device("cpu"))
                model.eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                with torch.no_grad():
                    p = model.predict_log_probs(torch.randint(2048, (1, 256)))
                self.assertTrue(torch.isfinite(p).all())
                torch.testing.assert_close(p.logsumexp(-1), torch.zeros(1, 256), atol=2e-6, rtol=0)

    def test_causality_short_windows_and_independence(self):
        for kind in ("prefix_copy", "mos"):
            model = build_model(self.config(kind)).eval()
            x = torch.randint(2048, (2, 17))
            changed = x.clone()
            changed[:, 9:] = (changed[:, 9:] + 23) % 2048
            with torch.no_grad():
                p = model.predict_log_probs(x)
                q = model.predict_log_probs(changed)
                alone = model.predict_log_probs(x[:1])
                short = model.predict_log_probs(x[:, :9])
                model.predict_log_probs(torch.randint(2048, (3, 8)))
                repeated = model.predict_log_probs(x)
                one = model.predict_log_probs(x[:, :1])
            torch.testing.assert_close(p[:, :9], q[:, :9], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(p[:, :9], short, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p[:1], alone, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p, repeated, atol=0, rtol=0)
            torch.testing.assert_close(one.logsumexp(-1), torch.zeros(2, 1), atol=1e-6, rtol=0)

    def test_copy_support_is_exactly_observed_prefix(self):
        model = build_model(self.config("prefix_copy")).eval()
        x = torch.tensor([[4, 7, 4, 9]])
        with torch.no_grad():
            model.copy_query.weight.zero_()
            model.copy_key.weight.zero_()
            p = model.copy_distribution(model.features(x), x)
        expected = torch.zeros(1, 4, 2048)
        for t in range(4):
            for j in range(t + 1):
                expected[0, t, x[0, j]] += 1 / (t + 1)
        torch.testing.assert_close(p, expected)

    def test_all_parameters_receive_finite_gradients(self):
        for kind in ("prefix_copy", "mos"):
            model = build_model(self.config(kind)).train()
            # Repetitions ensure copy-attention parameters receive useful gradients.
            x = torch.tensor([[4, 7, 4, 9, 7, 4], [8, 3, 8, 2, 3, 8]])
            logp = model(x[:, :-1])
            loss = F.cross_entropy(logp.flatten(0, 1), x[:, 1:].flatten())
            direct = F.nll_loss(logp.flatten(0, 1), x[:, 1:].flatten())
            torch.testing.assert_close(loss, direct, atol=1e-6, rtol=1e-6)
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_extreme_copy_gates_remain_finite_and_normalized(self):
        model = build_model(self.config("prefix_copy")).eval()
        x = torch.tensor([[1, 1, 2, 3]])
        for bias in (-80., 80.):
            with torch.no_grad():
                model.copy_gate.bias.fill_(bias)
                p = model(x)
            self.assertTrue(torch.isfinite(p).all())
            torch.testing.assert_close(p.logsumexp(-1), torch.zeros(1, 4), atol=2e-6, rtol=0)

    def test_serialization_and_no_hidden_state(self):
        for kind in ("prefix_copy", "mos"):
            config = self.config(kind)
            model = build_model(config).eval()
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = build_model(config).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            x = torch.randint(2048, (2, 13))
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for BF16 training check")
    def test_cuda_autocast_backward_and_fp32_evaluation(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 is unavailable")
        for kind in ("prefix_copy", "mos"):
            model = build_model(self.config(kind)).cuda().train()
            x = torch.randint(16, (2, 257), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp = model(x[:, :-1])
                loss = F.cross_entropy(logp.flatten(0, 1), x[:, 1:].flatten())
            self.assertEqual(logp.dtype, torch.float32)
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            model.eval()
            with torch.no_grad():
                logp = model(x[:, :-1])
            self.assertTrue(torch.isfinite(logp).all())
            torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, 256, device="cuda"),
                                       atol=2e-6, rtol=0)

    def test_mos_matches_explicit_probability_mixture(self):
        model = build_model(self.config("mos")).eval()
        x = torch.randint(2048, (2, 9))
        with torch.no_grad():
            hidden = model.features(x)
            z = model.mos_projection(hidden).view(2, 9, 2, 32).tanh()
            expected = (model.head(z).softmax(-1)
                        * model.mos_gate(hidden).softmax(-1).unsqueeze(-1)).sum(-2)
            torch.testing.assert_close(model(x).exp(), expected, atol=1e-8, rtol=1e-5)

    def test_dependency_guard_and_invalid_options(self):
        with patch("student_structured.BACKBONE_SHA256", "changed"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_model(self.config("mos"))
        with self.assertRaises(ValueError):
            build_model(self.config("unknown"))
        with self.assertRaises(ValueError):
            build_model(self.config("mos") | {"mos_components": 0})


if __name__ == "__main__":
    unittest.main()
