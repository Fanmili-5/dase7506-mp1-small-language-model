"""Fixed Stage212 local-window causality, export and gradient checks."""
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

import student_stage212_sliding_local as local
import student_stage212_sliding_local_rdrop as training
import student_stage212_sliding_local_structured as deployed


ROOT = Path(__file__).resolve().parents[1]


class Stage212SlidingLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config = json.loads((ROOT / "configs/stage212_sliding_local_attention_rdrop.json")
                                .read_text(encoding="utf-8"))

    def test_local_block_matches_explicit_causal_window(self):
        config = dict(self.config, width=32, heads=4, mlp_ratio=2.0, dropout=0.0)
        torch.manual_seed(17)
        block = local.SlidingLocalAttentionBlock(config).eval()
        x = torch.randn(2, 13, 32)
        with torch.inference_mode():
            actual = block(x)
            q, k, v = (block.qkv(block.norm1(x))
                       .view(2, 13, 3, 4, 8).permute(2, 0, 3, 1, 4))
            q, k = block.rope(q, k)
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (8 ** .5)
            positions = torch.arange(13)
            allowed = (positions[None, :] <= positions[:, None]) & (
                positions[None, :] >= positions[:, None] - 6)
            weights = F.softmax(scores.masked_fill(~allowed, -1.0e9), dim=-1)
            attended = torch.matmul(weights, v.float())
            attended = attended.transpose(1, 2).reshape(2, 13, 32)
            mixed = x + block.proj(attended)
            expected = mixed + block.mlp(block.norm2(mixed))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_training_to_inference_export_is_exact_and_causal(self):
        torch.manual_seed(17)
        source = training.build_model(self.config).eval()
        target = deployed.build_model(training.inference_config(self.config)).eval()
        target.load_state_dict(training.inference_state(source.state_dict()), strict=True)
        ids = torch.randint(2048, (2, 20))
        with torch.inference_mode():
            expected = source.predict_log_probs(ids)
            actual = target.predict_log_probs(ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual.logsumexp(-1),
                                       torch.zeros_like(actual[..., 0]),
                                       atol=2e-6, rtol=0)
            changed = ids.clone()
            changed[0, -1] = (changed[0, -1] + 1) % 2048
            torch.testing.assert_close(target.predict_log_probs(changed)[0, :-1],
                                       actual[0, :-1], atol=2e-6, rtol=0)
            torch.testing.assert_close(target.predict_log_probs(ids[:1])[0],
                                       actual[0], atol=2e-6, rtol=0)

    def test_rdrop_local_qkv_has_finite_gradients(self):
        torch.manual_seed(17)
        model = training.build_model(self.config).train()
        ids = torch.randint(2048, (2, 16))
        labels = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (2, *ids.shape))
        loss, _ = model.rdrop_training_loss(ids, labels, future)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for index in (1, 3, 5, 7):
            grad = model.blocks[index].qkv.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
