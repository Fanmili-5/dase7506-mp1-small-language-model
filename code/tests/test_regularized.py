"""Contract and training tests on synthetic tokens; never experiment weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

from common import make_model
import student_regularized as regularized
from scripts.export_regularized import export_payload
from train_experiment import checkpoint_payload


class RegularizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, row=.1, hidden=.2):
        return dict(vocab=2048, context=256, width=16, heads=2, depth=2,
            position="rope", activation="swiglu", norm="rmsnorm", dropout=.1,
            output_kind="prefix_copy", copy_dim=8, embedding_row_dropout=row,
            ffn_hidden_dropout=hidden)

    def test_initialization_rng_and_eval_are_identical_to_original(self):
        for config in (self.config(.1, 0), self.config(0, .2)):
            torch.manual_seed(17)
            original, _ = make_model("student_structured", regularized.inference_config(config), torch.device("cpu"))
            rng = torch.get_rng_state()
            torch.manual_seed(17)
            model = regularized.build_model(config)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(original.state_dict().keys(), model.state_dict().keys())
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, original.state_dict()[key]))
            x = torch.arange(33)[None]
            with torch.no_grad():
                torch.testing.assert_close(original.eval()(x), model.eval()(x), atol=0, rtol=0)

    def test_zero_regularization_is_exact_training_control(self):
        config = self.config(0, 0)
        model = regularized.build_model(config).train()
        original, _ = make_model("student_structured", regularized.inference_config(config), torch.device("cpu"))
        original.load_state_dict(model.state_dict())
        x = torch.arange(32)[None]
        torch.manual_seed(71)
        a = model(x)
        torch.manual_seed(71)
        b = original.train()(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_embedding_mask_is_by_row_and_does_not_mutate_output_weights(self):
        model = regularized.build_model(self.config(.5, 0)).train()
        weights = model.token.weight.detach().clone()
        x = torch.arange(100).repeat(2, 1)
        values = model.input_embeddings(x)
        torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
        dropped = (values[0] == 0).all(-1)
        self.assertTrue(dropped.any() and (~dropped).any())
        torch.testing.assert_close(values[0, ~dropped], weights[:100][~dropped] * 2)
        self.assertTrue(torch.equal(model.head.weight, weights))
        self.assertIs(model.head.weight, model.token.weight)

    def test_causality_independence_normalization_and_export(self):
        config = self.config()
        model = regularized.build_model(config).eval()
        x = torch.arange(34).reshape(2, 17)
        changed = x.clone()
        changed[:, 9:] += 91
        with torch.no_grad():
            p = model(x)
            torch.testing.assert_close(p[:, :9], model(changed)[:, :9], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(p[:1], model(x[:1]), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p.logsumexp(-1), torch.zeros(2, 17), atol=1e-6, rtol=0)
            model(changed)
            torch.testing.assert_close(p, model(x), atol=0, rtol=0)
        payload = checkpoint_payload(model, "student_regularized", config, 17, 512)
        exported = export_payload(payload, "synthetic-test-sha")
        self.assertEqual(exported["implementation"], "student_structured")
        self.assertNotIn("embedding_row_dropout", exported["config"])
        self.assertEqual(exported["train_tokens"], 512)
        for key in payload["model"]:
            self.assertTrue(torch.equal(payload["model"][key], exported["model"][key]))

    def test_finite_gradients_and_exact_resume(self):
        for config in (self.config(.1, 0), self.config(0, .2)):
            model = regularized.build_model(config).train()
            x = torch.arange(64).reshape(2, 32) % 16
            F.cross_entropy(model(x).flatten(0, 1), x.flatten()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        from test_training_resume import TrainingResumeTests
        helper = TrainingResumeTests()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config()))
            full = helper.arguments(root / "full", config_path, implementation="student_regularized")
            partial = helper.arguments(root / "partial", config_path, implementation="student_regularized", stop_after_step=2)
            helper.run_training(full)
            helper.run_training(partial)
            partial.resume, partial.stop_after_step = True, 0
            helper.run_training(partial)
            a = torch.load(full.run_dir / "checkpoint.pt", weights_only=True)["model"]
            b = torch.load(partial.run_dir / "checkpoint.pt", weights_only=True)["model"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_invalid_probability_and_source_guard(self):
        for value in (-.1, 1., float("nan")):
            with self.assertRaises(ValueError):
                regularized.build_model(self.config(value, 0))
        with patch.object(regularized, "STRUCTURED_SHA", "bad"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                regularized.build_model(self.config())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bf16_gradients(self):
        for config in (self.config(.1, 0), self.config(0, .2)):
            model = regularized.build_model(config).cuda().train()
            x = torch.arange(256, device="cuda")[None] % 32
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).flatten(0, 1), x.flatten())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
