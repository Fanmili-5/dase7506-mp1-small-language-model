"""Stage154 exploratory model: causal normalized output and branch gradients."""
from __future__ import annotations

import unittest

import torch

import student_stage154_shared_last_block


CONFIG = {
    "vocab": 2048, "width": 64, "heads": 4, "depth": 8, "context": 256,
    "position": "rope", "norm": "rmsnorm", "norm_eps": 1e-5,
    "activation": "swiglu", "mlp_ratio": 2.5, "dropout": 0.0,
    "bias": False, "scaled_residual_init": True,
    "output_kind": "prefix_copy", "copy_dim": 16, "copy_gate_bias": -2.0,
    "embedding_row_dropout": 0.0, "ffn_hidden_dropout": 0.0,
    "deep_supervision_layers": [4, 6], "deep_supervision_weight": 0.2,
    "future_prediction_offsets": [2, 3], "future_prediction_weight": 0.2,
    "conv_layers": [2, 4, 6, 8], "conv_kernel": 3,
    "rdrop_alpha": 0.5, "branch_b_layers": ["attention"],
    "branch_mixture": 0.5,
}


class SharedLastBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model = student_stage154_shared_last_block.build_model(CONFIG)

    def test_normalized_causal_independent(self):
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        changed_future = torch.tensor([[1, 2, 3, 4, 5, 90, 91, 92]])
        unrelated = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]])
        with torch.inference_mode():
            a = self.model.predict_log_probs(first)
            b = self.model.predict_log_probs(changed_future)
            together = self.model.predict_log_probs(torch.cat((first, unrelated)))
            feature = self.model.features(first)
        self.assertEqual(tuple(a.shape), (1, 8, 2048))
        self.assertEqual(tuple(feature.shape), (1, 8, 128))
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(float(a.logsumexp(-1).abs().max()), 1e-5)
        self.assertTrue(torch.allclose(a[:, :5], b[:, :5], atol=1e-6))
        self.assertTrue(torch.allclose(a, together[:1], atol=1e-6))

    def test_both_branches_receive_finite_gradients(self):
        self.model.train()
        ids = torch.randint(0, 2048, (2, 8))
        targets = torch.randint(0, 2048, (2, 8))
        future = torch.randint(0, 2048, (2, 2, 8))
        loss, parts = self.model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.blocks[7].depthwise.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.branch_b.qkv.weight.grad).all())
        self.assertGreater(float(self.model.blocks[7].depthwise.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.branch_b.qkv.weight.grad.abs().sum()), 0)
        self.assertIn("symmetric_kl", parts)


if __name__ == "__main__":
    unittest.main()
