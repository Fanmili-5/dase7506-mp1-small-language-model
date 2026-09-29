import io
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

import student_multi_token
import student_successor_multi_token
import student_successor_only


ROOT = Path(__file__).resolve().parents[1]


class SuccessorOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def config(training=False):
        config = dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                      position="rope", norm="rmsnorm", activation="swiglu",
                      mlp_ratio=2.6666667, bias=False, dropout=.1,
                      output_kind="prefix_copy", copy_semantics="successor",
                      copy_dim=8, copy_gate_bias=-2.)
        if training:
            config.update(embedding_row_dropout=.1, ffn_hidden_dropout=0.,
                          deep_supervision_layers=[1], deep_supervision_weight=.2,
                          future_prediction_offsets=[2, 3], future_prediction_weight=.2)
        return config

    def test_uniform_alignment_and_empty_first_route(self):
        model = student_successor_only.build_model(self.config()).eval()
        ids = torch.tensor([[4, 7, 4, 9]])
        with torch.no_grad():
            model.copy_query.weight.zero_(); model.copy_key.weight.zero_()
            distribution = model.copy_distribution(model.features(ids), ids)
        expected = torch.zeros(1, 4, 2048)
        for t in range(1, 4):
            for j in range(t):
                expected[0, t, ids[0, j + 1]] += 1 / t
        torch.testing.assert_close(distribution, expected)

    def test_causal_normalized_reentrant_and_serializable(self):
        model = student_successor_only.build_model(self.config()).eval()
        ids = torch.randint(2048, (2, 256))
        changed = ids.clone(); changed[:, 17:] = (changed[:, 17:] + 1) % 2048
        with torch.no_grad():
            p = model(ids); q = model(changed); short = model(ids[:, :17]); again = model(ids)
        self.assertTrue(torch.isfinite(p).all())
        torch.testing.assert_close(p.logsumexp(-1), torch.zeros(2, 256), atol=2e-6, rtol=0)
        torch.testing.assert_close(p[:, :17], q[:, :17], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(p[:, :17], short, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(p, again, atol=0, rtol=0)
        stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
        restored = student_successor_only.build_model(self.config()).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(p, restored(ids), atol=0, rtol=0)

    def test_training_loss_has_finite_gradients(self):
        model = student_successor_multi_token.build_model(self.config(training=True)).train()
        extended = torch.randint(2048, (2, 8))
        ids, targets = extended[:, :5], extended[:, 1:6]
        future = torch.stack((extended[:, 2:7], extended[:, 3:8]))
        loss, parts = model.training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"primary", "deep", "future"})
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_full_config_parameter_count_matches_stage26(self):
        successor = json.loads((ROOT / "configs/stage40_successor_multi_token.json").read_text())
        content = json.loads((ROOT / "configs/stage26_multi_token.json").read_text())
        a = student_successor_multi_token.build_model(successor)
        b = student_multi_token.build_model(content)
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))


if __name__ == "__main__":
    unittest.main()
