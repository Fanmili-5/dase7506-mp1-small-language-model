import json
from pathlib import Path
import unittest

import torch

import student_moe_multi_token
import student_moe_structured
import student_multi_token

ROOT = Path(__file__).resolve().parents[1]


class MoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage46_top1_moe.json").read_text())

    def test_capacity_and_active_hidden_budget(self):
        dense = student_multi_token.build_model(self.config)
        routed = student_moe_multi_token.build_model(self.config)
        self.assertGreater(sum(p.numel() for p in routed.parameters()),
                           sum(p.numel() for p in dense.parameters()))
        for block in routed.blocks:
            self.assertEqual(len(block.mlp.experts), 2)
            self.assertEqual(block.mlp.experts[0].input.out_features // 2, 512)

    def test_training_gradients_and_export_equivalence(self):
        torch.manual_seed(17)
        training = student_moe_multi_token.build_model(self.config)
        ids = torch.randint(2048, (2, 24))
        targets = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (2, *ids.shape))
        training.train()
        loss, parts = training.training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("router_balance", parts)
        for block in training.blocks:
            self.assertIsNotNone(block.mlp.router.weight.grad)
            self.assertTrue(torch.isfinite(block.mlp.router.weight.grad).all())
        deployed_config = student_moe_multi_token.inference_config(self.config)
        deployed = student_moe_structured.build_model(deployed_config)
        deployed.load_state_dict(student_moe_multi_token.inference_state(training.state_dict()))
        training.eval(); deployed.eval()
        with torch.inference_mode():
            expected = training(ids)
            actual = deployed(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
                                   atol=2e-6, rtol=0)

    def test_causality(self):
        model = student_moe_structured.build_model(
            student_moe_multi_token.inference_config(self.config)).eval()
        first = torch.randint(2048, (1, 32))
        second = first.clone(); second[:, 17:] = torch.randint(2048, (1, 15))
        with torch.inference_mode():
            a = model(first)
            b = model(second)
        torch.testing.assert_close(a[:, :17], b[:, :17], atol=2e-6, rtol=0)

    def test_autocast_dispatch_dtype(self):
        model = student_moe_multi_token.build_model(self.config).train()
        ids = torch.randint(2048, (2, 16))
        targets = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (2, *ids.shape))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss, _ = model.training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))

    def test_dense_training_and_sparse_eval_dispatch_agree(self):
        torch.manual_seed(23)
        module = student_moe_structured.Top1SwiGLU(32, 48, 2, False, 2.0)
        inputs = torch.randn(3, 11, 32)
        module.train(); dense = module(inputs)
        module.eval(); sparse = module(inputs)
        torch.testing.assert_close(dense, sparse, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
