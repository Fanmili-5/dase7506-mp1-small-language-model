"""Stage186 structural, gradient, inference and causal guards."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_stage186_narrow_parallel_rdrop as stage186
import student_stage186_narrow_parallel_structured as inference_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage186_narrow_parallel_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"


class NarrowParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.control = json.loads(CONTROL.read_text(encoding="utf-8"))
        torch.set_num_threads(2)

    def test_fixed_config_zero_start_weights_rng_and_predictions(self):
        self.assertEqual({k: v for k, v in self.config.items()
                          if k not in {"parallel_heads", "parallel_head_dim"}}, self.control)
        torch.manual_seed(17)
        original = student_hybrid_conv_rdrop.build_model(self.control)
        original_rng = torch.get_rng_state()
        torch.manual_seed(17)
        narrow = stage186.build_model(self.config)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        for name, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(tensor, narrow.state_dict()[name]), name)
        self.assertGreater(sum(p.numel() for p in narrow.parameters()),
                           sum(p.numel() for p in original.parameters()))
        ids = torch.randint(0, 2048, (2, 16))
        original.eval(); narrow.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(original.predict_log_probs(ids),
                                        narrow.predict_log_probs(ids)))

    def test_each_narrow_branch_receives_gradient(self):
        torch.manual_seed(18)
        model = stage186.build_model(self.config).train()
        ids = torch.randint(0, 2048, (1, 8))
        targets = torch.randint(0, 2048, (1, 8))
        future = torch.randint(0, 2048, (2, 1, 8))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        loss.backward()
        for layer in self.config["conv_layers"]:
            grad = model.blocks[layer - 1].attn_proj.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_strict_inference_export_causality_and_normalization(self):
        torch.manual_seed(19)
        training = stage186.build_model(self.config).eval()
        inference = inference_module.build_model(
            stage186.inference_config(self.config)).eval()
        inference.load_state_dict(stage186.inference_state(training.state_dict()), strict=True)
        ids = torch.randint(0, 2048, (2, 16))
        changed = ids.clone(); changed[0, 12:] = 42
        with torch.inference_mode():
            train_logp = training.predict_log_probs(ids)
            inference_logp = inference.predict_log_probs(ids)
            mutated_logp = inference.predict_log_probs(changed)
        self.assertTrue(torch.equal(train_logp, inference_logp))
        self.assertLess(float((mutated_logp[0, :12] - inference_logp[0, :12]).abs().max()),
                        1e-5)
        self.assertTrue(torch.equal(mutated_logp[1], inference_logp[1]))
        self.assertLess(float(torch.logsumexp(inference_logp, dim=-1).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
