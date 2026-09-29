"""Synthetic invariants for Stage206's within-window linear memory."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import student_stage206_linear_memory_rdrop as stage206


class LinearMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage206_linear_memory_rdrop.json")
                                .read_text(encoding="utf-8"))
        torch.manual_seed(206017)
        torch.set_num_threads(4)

    def test_fixed_replacement_and_probability_contract(self):
        model = stage206.build_model(self.config).eval()
        self.assertEqual(model.linear_memory_layers, (3, 7))
        self.assertEqual(model.conv_layers, (2, 4, 6, 8))
        self.assertIsInstance(model.blocks[2], stage206.LinearMemoryBlock)
        self.assertIsInstance(model.blocks[6], stage206.LinearMemoryBlock)
        ids = torch.randint(0, 2048, (2, 12))
        with torch.inference_mode():
            logp = model.predict_log_probs(ids)
            changed = ids.clone()
            changed[0, -1] = (changed[0, -1] + 1) % 2048
            changed_logp = model.predict_log_probs(changed)
            separate = model.predict_log_probs(ids[:1])
        self.assertTrue(torch.isfinite(logp).all())
        self.assertLess(float(logp.logsumexp(-1).abs().max()), 1e-5)
        self.assertLess(float((logp[0, :-1] - changed_logp[0, :-1]).abs().max()), 1e-5)
        self.assertLess(float((logp[0] - separate[0]).abs().max()), 1e-5)

    def test_linear_state_matches_explicit_causal_prefix(self):
        block = stage206.LinearMemoryBlock(self.config).eval()
        x = torch.randn(1, 5, self.config["width"])
        with torch.inference_mode():
            mixed = block(x)
            rows = [block(x[:, :length])[:, -1] for length in range(1, 6)]
        explicit = torch.stack(rows, dim=1)
        self.assertLess(float((mixed - explicit).abs().max()), 2e-5)

    def test_training_gradients_reach_linear_memory(self):
        model = stage206.build_model(self.config).train()
        ids = torch.randint(0, 2048, (1, 8))
        targets = torch.randint(0, 2048, (1, 8))
        future = torch.randint(0, 2048, (2, 1, 8))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.blocks[2].qk.weight.grad)
        self.assertGreater(float(model.blocks[2].qk.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
