import json
from pathlib import Path
import unittest

import torch

from common import make_model
from muon_pilot import HiddenMatrixMuon, newton_schulz_zeroth_power, partition_hidden_matrices


ROOT = Path(__file__).resolve().parents[1]


class MuonPilotTests(unittest.TestCase):
    def test_zeroth_power_finite_and_shape(self):
        for shape in ((4, 8), (8, 4), (4, 4)):
            output = newton_schulz_zeroth_power(torch.randn(shape))
            self.assertEqual(output.shape, shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_hidden_partition_excludes_embedding_and_output(self):
        config = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        model, _ = make_model("student_hybrid_conv_rdrop", config, torch.device("cpu"))
        hidden, other, names = partition_hidden_matrices(model)
        self.assertTrue(hidden)
        self.assertTrue(other)
        self.assertIn("token.weight", names["adamw"])
        self.assertNotIn("token.weight", names["muon"])
        self.assertTrue(all(name.startswith("blocks.") for name in names["muon"]))
        self.assertEqual(sum(param.numel() for param in hidden + other),
                         sum(param.numel() for param in model.parameters()))

    def test_step_updates_matrix_but_not_missing_gradient(self):
        trained = torch.nn.Parameter(torch.ones(4, 8))
        idle = torch.nn.Parameter(torch.ones(8, 4))
        optimizer = HiddenMatrixMuon([trained, idle], lr=0.02)
        trained.grad = torch.randn_like(trained)
        optimizer.step()
        self.assertFalse(torch.equal(trained, torch.ones_like(trained)))
        self.assertTrue(torch.equal(idle, torch.ones_like(idle)))
        self.assertEqual(len(optimizer.state), 1)


if __name__ == "__main__":
    unittest.main()
