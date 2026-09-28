"""Input-only Stage208 structure and probability checks."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from student_stage208_lexical_hierarchy import LexicalHierarchy, build_model


ROOT = Path(__file__).resolve().parents[1]


class Stage208LexicalHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/stage208_lexical_hierarchy_rdrop.json").read_text())

    def test_tree_probabilities_and_coverage(self):
        torch.manual_seed(208)
        tree = LexicalHierarchy(2048, 288)
        hidden = torch.randn(2, 3, 288)
        with torch.inference_mode():
            logp = tree(hidden)
        self.assertEqual(logp.shape, (2, 3, 2048))
        self.assertEqual(tree.path_nodes.shape, (2048, 11))
        self.assertEqual(int(torch.unique(tree.path_nodes).numel()), 2047)
        self.assertLess(float(logp.logsumexp(-1).abs().max()), 1e-5)

    def test_predictor_normalization_prefix_and_rows(self):
        torch.manual_seed(208)
        model = build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 8))
        with torch.inference_mode():
            original = model.predict_log_probs(ids)
            future = ids.clone()
            future[0, -1] = (future[0, -1] + 1) % 2048
            changed = model.predict_log_probs(future)
            singleton = model.predict_log_probs(ids[:1])
        self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float((original[0, :-1] - changed[0, :-1]).abs().max()), 1e-5)
        self.assertLess(float((original[0] - singleton[0]).abs().max()), 1e-5)

    def test_train_step_has_finite_gradients(self):
        torch.manual_seed(208)
        model = build_model(self.config).train()
        ids = torch.randint(0, 2048, (1, 8))
        targets = torch.randint(0, 2048, (1, 8))
        future = torch.randint(0, 2048, (2, 1, 8))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(model.lexical.node_weight.grad).all()))
        self.assertGreater(float(model.lexical.node_weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
