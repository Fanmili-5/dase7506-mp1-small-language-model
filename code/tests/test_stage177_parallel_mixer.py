"""Guard Stage177's zero-start, causality, gradients and inference export."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_stage177_parallel_mixer_rdrop as stage177
import student_stage177_parallel_mixer_structured as stage177_inference


CONFIG = Path(__file__).resolve().parents[1] / "configs/stage177_parallel_mixer_rdrop.json"


class TestParallelMixer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        torch.set_num_threads(2)

    def test_zero_start_and_rng_match_stage54(self):
        torch.manual_seed(17)
        original = student_hybrid_conv_rdrop.build_model(self.config)
        original_rng = torch.get_rng_state()
        torch.manual_seed(17)
        parallel = stage177.build_model(self.config)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        parent_state = original.state_dict()
        child_state = parallel.state_dict()
        for name, tensor in parent_state.items():
            self.assertTrue(torch.equal(tensor, child_state[name]), name)
        ids = torch.randint(0, 2048, (1, 16))
        original.eval(); parallel.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(original.predict_log_probs(ids),
                                        parallel.predict_log_probs(ids)))

    def test_parallel_branch_receives_gradient(self):
        torch.manual_seed(18)
        model = stage177.build_model(self.config)
        model.train()
        ids = torch.randint(0, 2048, (1, 8))
        targets = torch.randint(0, 2048, (1, 8))
        future = torch.randint(0, 2048, (2, 1, 8))
        loss, _ = model.rdrop_training_loss(ids, targets, future)
        loss.backward()
        for layer in self.config["conv_layers"]:
            grad = model.blocks[layer - 1].attn_proj.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_inference_export_and_causality(self):
        torch.manual_seed(19)
        training = stage177.build_model(self.config).eval()
        inference = stage177_inference.build_model(
            stage177.inference_config(self.config)).eval()
        inference.load_state_dict(stage177.inference_state(training.state_dict()), strict=True)
        ids = torch.randint(0, 2048, (2, 16))
        changed = ids.clone(); changed[0, 12:] = 42
        with torch.inference_mode():
            train_logp = training.predict_log_probs(ids)
            inference_logp = inference.predict_log_probs(ids)
            self.assertTrue(torch.equal(train_logp, inference_logp))
            mutated_logp = inference.predict_log_probs(changed)
        self.assertLess(float((mutated_logp[0, :12] - inference_logp[0, :12]).abs().max()),
                        1e-5)
        self.assertTrue(torch.equal(mutated_logp[1], inference_logp[1]))
        self.assertLess(float(torch.logsumexp(inference_logp, dim=-1).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
