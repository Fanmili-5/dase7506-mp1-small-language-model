"""Zero-delta, gradient, and exact-merge tests for the Stage129 intervention."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import student_hybrid_conv_output_bias
from scripts.analyze_stage98_distilled_gate import distilled_log_probs
from scripts.train_stage129_backbone_lora import attach_lora, merged_model


class Stage129BackboneLoraTests(unittest.TestCase):
    def test_zero_delta_gradients_and_merge(self):
        torch.set_num_threads(2)
        torch.manual_seed(129017)
        config = json.loads((ROOT / "configs/stage69_hybrid_conv_output_lora.json")
                            .read_text(encoding="utf-8"))
        config.pop("output_lora_rank")
        config.pop("output_lora_alpha")
        model = student_hybrid_conv_output_bias.build_model(config).eval()
        ids = torch.tensor([[3, 11, 7, 23], [5, 2, 13, 17]])
        prior = torch.zeros(config["vocab"])
        with torch.no_grad():
            original = distilled_log_probs(model, ids, prior)
        names, trainable = attach_lora(model)
        self.assertEqual(len(names), 32)
        self.assertEqual(len(trainable), 64)
        with torch.no_grad():
            initial = distilled_log_probs(model, ids, prior)
        torch.testing.assert_close(original, initial, atol=1e-6, rtol=1e-6)
        loss = -distilled_log_probs(model, ids, prior)[..., 1].mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            and parameter.grad.abs().sum() > 0
                            for parameter in trainable))
        optimizer = torch.optim.AdamW(trainable, lr=.002)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            adapted = distilled_log_probs(model, ids, prior)
            merged = merged_model(model)
            after_merge = distilled_log_probs(merged, ids, prior)
            adapted_again = distilled_log_probs(model, ids, prior)
        torch.testing.assert_close(adapted, after_merge, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(adapted, adapted_again, atol=1e-6, rtol=1e-6)
        self.assertFalse(any("parametrizations" in key for key in merged.state_dict()))


if __name__ == "__main__":
    unittest.main()
