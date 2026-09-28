"""Stage209 exact tree-probability equivalence and causal model tests."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from student_stage208_lexical_hierarchy import LexicalHierarchy
from student_stage209_tree_propagation import PropagatedLexicalHierarchy, build_model


ROOT = Path(__file__).resolve().parents[1]


class Stage209TreePropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/stage208_lexical_hierarchy_rdrop.json").read_text())

    def test_exact_original_tree_formula(self):
        torch.manual_seed(209)
        old = LexicalHierarchy(2048, 288).eval()
        new = PropagatedLexicalHierarchy(2048, 288).eval()
        with torch.no_grad():
            old.node_weight.normal_(0, .02)
            old.node_bias.normal_(0, .02)
            new.node_weight.copy_(old.node_weight)
            new.node_bias.copy_(old.node_bias)
        hidden = torch.randn(2, 9, 288)
        with torch.inference_mode():
            old_logp = old(hidden)
            new_logp = new(hidden)
        self.assertLess(float((old_logp - new_logp).abs().max()), 2e-6)
        self.assertLess(float(new_logp.logsumexp(-1).abs().max()), 1e-5)

    def test_predictor_causal_and_finite_training_gradients(self):
        torch.manual_seed(209)
        model = build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 8))
        with torch.inference_mode():
            original = model.predict_log_probs(ids)
            changed = ids.clone()
            changed[0, -1] = (changed[0, -1] + 1) % 2048
            future = model.predict_log_probs(changed)
            one_row = model.predict_log_probs(ids[:1])
        self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float((original[0, :-1] - future[0, :-1]).abs().max()), 1e-5)
        self.assertLess(float((original[0] - one_row[0]).abs().max()), 1e-5)
        model.train()
        loss, _ = model.rdrop_training_loss(
            ids[:1], torch.randint(0, 2048, (1, 8)),
            torch.randint(0, 2048, (2, 1, 8)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(model.lexical.node_weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
