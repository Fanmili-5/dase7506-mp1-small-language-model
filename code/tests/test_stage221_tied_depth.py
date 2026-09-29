"""Structural and causal tests for the shared-tail-pair Transformer."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

import student_stage221_tied_depth_rdrop as training
import student_stage221_tied_depth_structured as inference

ROOT = Path(__file__).resolve().parents[1]


class Stage221Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/stage221_tied_depth_rdrop.json").read_text())

    def test_shared_pair_and_exact_export(self):
        torch.manual_seed(17)
        model = training.build_model(self.config).eval()
        self.assertIs(model.blocks[8], model.blocks[6])
        self.assertIs(model.blocks[9], model.blocks[7])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8_106_049)
        deployed = inference.build_model(training.inference_config(self.config)).eval()
        deployed.load_state_dict(training.inference_state(model.state_dict()), strict=True)
        ids = torch.randint(0, 2048, (2, 16), generator=torch.Generator().manual_seed(221))
        with torch.inference_mode():
            torch.testing.assert_close(model.predict_log_probs(ids),
                                       deployed.predict_log_probs(ids), atol=0, rtol=0)

    def test_causal_normalized_and_independent(self):
        torch.manual_seed(17)
        model = training.build_model(self.config).eval()
        ids = torch.randint(0, 2048, (2, 16), generator=torch.Generator().manual_seed(2210))
        with torch.inference_mode():
            original = model.predict_log_probs(ids)
            self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
            changed = ids.clone()
            changed[:, 8:] = (changed[:, 8:] + 97) % 2048
            self.assertLess(float((original[:, :8]
                                   - model.predict_log_probs(changed)[:, :8]).abs().max()), 3e-5)
            self.assertLess(float((original[:1]
                                   - model.predict_log_probs(ids[:1])).abs().max()), 1e-5)

    def test_shared_pair_receives_gradients(self):
        torch.manual_seed(17)
        model = training.build_model(self.config).train()
        ids = torch.randint(0, 2048, (2, 16), generator=torch.Generator().manual_seed(2211))
        targets = torch.roll(ids, shifts=-1, dims=1)
        future = torch.stack((torch.roll(ids, -2, 1), torch.roll(ids, -3, 1)))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.blocks[6].qkv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.blocks[7].depthwise.weight.grad.abs().sum()), 0)

    def test_rejects_altered_sharing_contract(self):
        config = dict(self.config, shared_tail_pair=[5, 6])
        with self.assertRaises(ValueError):
            training.build_model(config)


if __name__ == "__main__":
    unittest.main()
