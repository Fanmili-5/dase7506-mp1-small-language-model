"""Structural checks for training-only SAM; no validation or test text."""
from __future__ import annotations

import unittest
import json
from pathlib import Path

import torch

from common import make_model
from scripts.sam_training_step import sam_training_step


class SAMTrainingStepTests(unittest.TestCase):
    def test_stage54_initial_state_and_inference_are_unchanged(self):
        config = json.loads((Path(__file__).resolve().parents[1]
                             / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        torch.manual_seed(17)
        reference, _ = make_model("student_hybrid_conv_rdrop", config,
                                  torch.device("cpu"))
        torch.manual_seed(17)
        candidate, _ = make_model("student_hybrid_conv_rdrop", config,
                                  torch.device("cpu"))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        reference.eval()
        candidate.eval()
        ids = torch.arange(256).remainder(config["vocab"]).unsqueeze(0)
        with torch.no_grad():
            self.assertTrue(torch.equal(reference.predict_log_probs(ids),
                                        candidate.predict_log_probs(ids)))

    def make_model(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -1.0]]))
        return model

    def test_perturbation_has_fixed_radius_and_original_is_restored(self):
        model = self.make_model()
        original = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        seen = {}

        def ascent():
            loss = (model(torch.tensor([[1.0, 2.0]])) - 3.0).square().mean()
            return loss, {}

        def descent():
            seen["perturbed"] = model.weight.detach().clone()
            loss = (model(torch.tensor([[2.0, -1.0]])) - 1.0).square().mean()
            return loss, {}

        result = sam_training_step(model, optimizer, ascent, descent, rho=0.05)
        self.assertTrue(torch.equal(model.weight, original))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(seen["perturbed"] - original)),
            0.05, places=6)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(result["ascent_gradient_norm"], 0)

    def test_exception_restores_parameters_and_skips_update(self):
        model = self.make_model()
        original = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def ascent():
            return (model(torch.tensor([[1.0, 2.0]])) - 3.0).square().mean(), {}

        def descent():
            raise RuntimeError("deliberate test failure")

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            sam_training_step(model, optimizer, ascent, descent, rho=0.05)
        self.assertTrue(torch.equal(model.weight, original))

    def test_invalid_radius_is_rejected(self):
        model = self.make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaises(ValueError):
            sam_training_step(model, optimizer, lambda: (model.weight.sum(), {}),
                              lambda: (model.weight.sum(), {}), rho=0)


if __name__ == "__main__":
    unittest.main()
