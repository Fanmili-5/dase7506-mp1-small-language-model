"""Stage215 train-only basis and input-stream contract tests."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts.build_stage215_semantic_basis import (
    basis_from_counts, count_symmetric_contexts)
from student_hybrid_conv_rdrop import build_model as control_model
from student_stage215_semantic_input import build_model as candidate_model


ROOT = Path(__file__).resolve().parents[1]


class Stage215SemanticInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.control_config = json.loads(
            (ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        cls.candidate_config = json.loads(
            (ROOT / "configs/stage215_semantic_input_rdrop.json").read_text())

    def make_pair(self):
        torch.manual_seed(17)
        control = control_model(self.control_config).eval()
        torch.manual_seed(17)
        candidate = candidate_model(self.candidate_config).eval()
        basis = torch.arange(2048 * 64, dtype=torch.float32).reshape(2048, 64)
        basis = basis.remainder(131).div(131)
        candidate.set_semantic_basis(basis)
        return control, candidate

    def test_fixed_cooccurrence_is_symmetric_and_finite(self) -> None:
        ids = np.asarray([0, 1, 2, 1, 3, 2, 0], dtype=np.int64)
        counts = count_symmetric_contexts(ids, vocab=4, distance=2)
        np.testing.assert_array_equal(counts, counts.T)
        self.assertGreater(counts[0, 1], 0)
        basis = basis_from_counts(counts, dim=2)
        self.assertEqual(tuple(basis.shape), (4, 2))
        self.assertTrue(torch.isfinite(basis).all())

    def test_zero_start_same_shared_weights_and_predictions(self) -> None:
        control, candidate = self.make_pair()
        original = control.state_dict()
        changed = candidate.state_dict()
        for name, tensor in original.items():
            torch.testing.assert_close(tensor, changed[name], rtol=0, atol=0)
        ids = torch.randint(0, 2048, (2, 12))
        with torch.inference_mode():
            a = control.predict_log_probs(ids)
            b = candidate.predict_log_probs(ids)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        torch.testing.assert_close(b.logsumexp(-1), torch.zeros(2, 12),
                                   rtol=0, atol=1e-6)

    def test_nonzero_semantic_path_is_causal_and_receives_gradients(self) -> None:
        _, candidate = self.make_pair()
        with torch.no_grad():
            candidate.semantic_projection.weight.fill_(0.005)
        ids = torch.randint(0, 2048, (2, 12))
        altered = ids.clone()
        altered[0, 7:] = (altered[0, 7:] + 31) % 2048
        with torch.inference_mode():
            original = candidate.predict_log_probs(ids)
            changed = candidate.predict_log_probs(altered)
            alone = candidate.predict_log_probs(ids[:1])
        torch.testing.assert_close(original[0, :7], changed[0, :7],
                                   rtol=0, atol=1e-5)
        torch.testing.assert_close(original[:1], alone, rtol=0, atol=1e-5)
        candidate.train()
        candidate.zero_grad(set_to_none=True)
        logits = candidate.predict_log_probs(ids)
        targets = (ids + 1) % 2048
        loss = F.nll_loss(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradient = candidate.semantic_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
