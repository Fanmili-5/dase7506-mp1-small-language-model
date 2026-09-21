"""Contract, strict successor alignment, gradients and matched control tests."""
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import student_structured
import student_successor


class SuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, kind="dual_copy"):
        return dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                    position="rope", norm="rmsnorm", activation="swiglu",
                    mlp_ratio=2.6666667, bias=False, dropout=0.1,
                    output_kind="prefix_copy", copy_dim=8, experiment_kind=kind)

    def test_uniform_successor_support_and_alignment(self):
        model = student_successor.build_model(self.config()).eval()
        ids = torch.tensor([[4, 7, 4, 9]])
        with torch.no_grad():
            model.successor_query.weight.zero_()
            model.successor_key.weight.zero_()
            distribution = model.successor_distribution(model.features(ids), ids)
        expected = torch.zeros(1, 4, 2048)
        for t in range(1, 4):
            for j in range(t):
                expected[0, t, ids[0, j + 1]] += 1 / t
        torch.testing.assert_close(distribution, expected)

    def test_control_preserves_b_initialization_and_rng(self):
        torch.manual_seed(17)
        original = student_structured.build_model(self.config())
        original_rng = torch.get_rng_state()
        torch.manual_seed(17)
        control = student_successor.build_model(self.config("fp32_control"))
        torch.testing.assert_close(torch.get_rng_state(), original_rng, atol=0, rtol=0)
        for name, weight in control.state_dict().items():
            torch.testing.assert_close(weight, original.state_dict()[name], atol=0, rtol=0)
        self.assertFalse(any(name.startswith("copy_") for name, _ in control.named_parameters()))

    def test_causality_reentrancy_batch_and_normalization(self):
        for kind in ("dual_copy", "fp32_control"):
            model = student_successor.build_model(self.config(kind)).eval()
            ids = torch.randint(2048, (2, 256))
            changed = ids.clone()
            changed[:, 8:] = (changed[:, 8:] + 9) % 2048
            with torch.no_grad():
                p, q = model(ids), model(changed)
                short = model(ids[:, :8])
                one = model(ids[:, :1])
                alone = model(ids[:1])
                again = model(ids)
            self.assertTrue(torch.isfinite(p).all())
            torch.testing.assert_close(p.logsumexp(-1), torch.zeros(2, 256), atol=2e-6, rtol=0)
            torch.testing.assert_close(one.logsumexp(-1), torch.zeros(2, 1), atol=2e-6, rtol=0)
            torch.testing.assert_close(p[:, :8], q[:, :8], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(p[:, :8], short, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p[:1], alone, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p, again, atol=0, rtol=0)

    def test_finite_backward_and_reload(self):
        for kind in ("dual_copy", "fp32_control"):
            model = student_successor.build_model(self.config(kind)).train()
            x = torch.tensor([[4, 7, 4, 9, 7, 4], [8, 3, 8, 2, 3, 8]])
            p = model(x[:, :-1])
            loss = F.cross_entropy(p.flatten(0, 1), x[:, 1:].flatten())
            torch.testing.assert_close(loss, F.nll_loss(p.flatten(0, 1), x[:, 1:].flatten()))
            loss.backward()
            for name, weight in model.named_parameters():
                self.assertIsNotNone(weight.grad, name)
                self.assertTrue(torch.isfinite(weight.grad).all(), name)
                self.assertGreater(weight.grad.abs().sum().item(), 0, name)
            model.eval()
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = student_successor.build_model(self.config(kind)).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)

    def test_mixture_matches_explicit_probabilities(self):
        model = student_successor.build_model(self.config()).eval()
        ids = torch.tensor([[4, 7, 4, 9, 7, 4]])
        with torch.no_grad():
            h = model.features(ids)
            logits = torch.cat((torch.zeros(1, 6, 1), model.copy_gate(h), model.successor_gate(h)), -1)
            logits[:, 0, 2] = -float("inf")
            g = logits.softmax(-1)
            expected = (model.head(h).softmax(-1) * g[:, :, :1]
                        + model.copy_distribution(h, ids) * g[:, :, 1:2]
                        + model.successor_distribution(h, ids) * g[:, :, 2:])
            torch.testing.assert_close(model(ids).exp(), expected, atol=1e-7, rtol=1e-5)

    def test_dependency_guards(self):
        with patch("student_successor.STRUCTURED_SHA256", "bad"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                student_successor.build_model(self.config())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA BF16 requires Windows training host")
    def test_cuda_mixed_precision(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("No BF16 support")
        for kind in ("dual_copy", "fp32_control"):
            model = student_successor.build_model(self.config(kind)).cuda().train()
            x = torch.randint(16, (2, 257), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x[:, :-1])
                loss = F.cross_entropy(p.flatten(0, 1), x[:, 1:].flatten())
            self.assertEqual(p.dtype, torch.float32)
            loss.backward()
            for name, weight in model.named_parameters():
                self.assertTrue(torch.isfinite(weight.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
