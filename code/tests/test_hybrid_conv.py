import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_multi_token
import student_hybrid_conv_structured

ROOT = Path(__file__).resolve().parents[1]


class HybridConvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage49_hybrid_conv.json").read_text())

    def test_causality_and_normalization(self):
        model = student_hybrid_conv_structured.build_model(
            student_hybrid_conv_multi_token.inference_config(self.config)).eval()
        first = torch.randint(2048, (2, 48))
        second = first.clone(); second[:, 27:] = torch.randint(2048, (2, 21))
        with torch.inference_mode():
            a = model(first); b = model(second)
        torch.testing.assert_close(a[:, :27], b[:, :27], atol=2e-6, rtol=0)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros_like(a[..., 0]),
                                   atol=2e-6, rtol=0)

    def test_training_gradients_and_export_equivalence(self):
        torch.manual_seed(17)
        training = student_hybrid_conv_multi_token.build_model(self.config)
        ids = torch.randint(2048, (2, 24))
        targets = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (2, *ids.shape))
        training.train(); loss, _ = training.training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(training.blocks[1].depthwise.weight.grad).all())
        deployed_config = student_hybrid_conv_multi_token.inference_config(self.config)
        deployed = student_hybrid_conv_structured.build_model(deployed_config)
        deployed.load_state_dict(
            student_hybrid_conv_multi_token.inference_state(training.state_dict()), strict=True)
        training.eval(); deployed.eval()
        with torch.inference_mode():
            expected = training(ids); actual = deployed(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_local_heavy_layout_is_causal(self):
        training_config = json.loads(
            (ROOT / "configs/stage58_local_heavy_rdrop.json").read_text())
        config = student_hybrid_conv_multi_token.inference_config(training_config)
        model = student_hybrid_conv_structured.build_model(config).eval()
        self.assertEqual(model.conv_layers, (2, 3, 4, 6, 7, 8))
        self.assertEqual(model.token.embedding_dim, 320)
        first = torch.randint(2048, (1, 32))
        second = first.clone(); second[:, 19:] = torch.randint(2048, (1, 13))
        with torch.inference_mode():
            a = model(first); b = model(second)
        torch.testing.assert_close(a[:, :19], b[:, :19], atol=2e-6, rtol=0)

    def test_balanced_layout_is_causal(self):
        training_config = json.loads(
            (ROOT / "configs/stage70_balanced_hybrid_rdrop.json").read_text())
        config = student_hybrid_conv_multi_token.inference_config(training_config)
        model = student_hybrid_conv_structured.build_model(config).eval()
        self.assertEqual(model.conv_layers, (2, 3, 5, 6, 8))
        self.assertEqual(model.token.embedding_dim, 304)
        first = torch.randint(2048, (1, 32))
        second = first.clone(); second[:, 19:] = torch.randint(2048, (1, 13))
        with torch.inference_mode():
            a = model(first); b = model(second)
        torch.testing.assert_close(a[:, :19], b[:, :19], atol=2e-6, rtol=0)

    def test_resource_adjusted_balanced_layout_is_causal(self):
        training_config = json.loads(
            (ROOT / "configs/stage72_balanced_hybrid_rdrop.json").read_text())
        config = student_hybrid_conv_multi_token.inference_config(training_config)
        model = student_hybrid_conv_structured.build_model(config).eval()
        self.assertEqual(model.conv_layers, (2, 3, 5, 6, 8))
        self.assertEqual(model.token.embedding_dim, 300)
        self.assertEqual(model.blocks[0].heads, 6)
        first = torch.randint(2048, (1, 32))
        second = first.clone(); second[:, 19:] = torch.randint(2048, (1, 13))
        with torch.inference_mode():
            a = model(first); b = model(second)
        torch.testing.assert_close(a[:, :19], b[:, :19], atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
