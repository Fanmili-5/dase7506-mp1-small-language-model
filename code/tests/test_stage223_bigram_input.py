"""Structural and causal checks for train-derived bigram input."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from student_stage223_bigram_input import build_train_bigram_rank_map
import student_stage223_bigram_rdrop as training
import student_stage223_bigram_structured as inference


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/stage223_bigram_input_rdrop.json").read_text())


class Stage223Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_train_only_rank_map_is_deterministic(self) -> None:
        tokens = torch.randint(0, 2048, (50000,), generator=torch.Generator().manual_seed(223))
        first = build_train_bigram_rank_map(tokens)
        second = build_train_bigram_rank_map(tokens)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int((first > 0).sum()), 16384)
        self.assertEqual(int(first.max()), 16384)

    def test_train_export_state_and_pair_boundary(self) -> None:
        torch.manual_seed(223)
        model = training.build_model(CONFIG).eval()
        with torch.no_grad():
            model.bigram_rank_map[1 * 2048 + 2] = 17
            model.bigram_rank_map[2 * 2048 + 3] = 18
        ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        self.assertEqual(model.pair_ranks(ids).tolist(),
                         [[0, 17, 18, 0], [0, 18, 0, 0]])
        exported = inference.build_model(training.inference_config(CONFIG)).eval()
        exported.load_state_dict(training.inference_state(model.state_dict()), strict=True)
        with torch.inference_mode():
            difference = (model.predict_log_probs(ids)
                          - exported.predict_log_probs(ids)).abs().max()
        self.assertLess(float(difference), 1e-6)

    def test_normalized_causal_and_independent(self) -> None:
        torch.manual_seed(223)
        model = inference.build_model(training.inference_config(CONFIG)).eval()
        with torch.no_grad():
            model.bigram_rank_map[1 * 2048 + 2] = 17
            model.bigram_rank_map[2 * 2048 + 3] = 18
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        with torch.inference_mode():
            logp = model.predict_log_probs(ids)
            changed = ids.clone(); changed[:, 4:] += 19
            future_error = (logp[:, :4]
                            - model.predict_log_probs(changed)[:, :4]).abs().max()
            row_error = (logp[:1]
                         - model.predict_log_probs(ids[:1])).abs().max()
        self.assertLess(float(logp.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float(future_error), 1e-5)
        self.assertLess(float(row_error), 1e-5)

    def test_bigram_parameters_receive_gradient(self) -> None:
        torch.manual_seed(223)
        model = training.build_model(CONFIG).train()
        with torch.no_grad():
            model.bigram_rank_map[1 * 2048 + 2] = 17
        ids = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2]])
        targets = torch.tensor([[2, 1, 2, 3], [2, 1, 2, 3]])
        future = torch.stack((targets, targets))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.bigram_embedding.weight.grad[17].norm()), 0)
        self.assertGreater(float(model.bigram_projection.weight.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
