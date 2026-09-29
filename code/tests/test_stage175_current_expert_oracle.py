"""Guard the target-only expert recovery used in Stage175 diagnostics."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_stage175_current_expert_oracle import recover_target_probabilities


class TestCurrentExpertOracle(unittest.TestCase):
    def test_recovers_two_experts_without_using_full_distributions(self):
        neural = torch.tensor([0.30, 0.01], dtype=torch.float32)
        count = torch.tensor([0.10, 0.20], dtype=torch.float32)
        weight = torch.tensor([0.25, 0.75], dtype=torch.float32)
        before = (1 - weight) * neural
        after = before + weight * count
        recovered_neural, recovered_count = recover_target_probabilities(
            before, after, weight)
        self.assertTrue(torch.allclose(recovered_neural, neural.double(), atol=1e-7))
        self.assertTrue(torch.allclose(recovered_count, count.double(), atol=1e-7))
        self.assertEqual(int((recovered_count > recovered_neural).sum()), 1)

    def test_rejects_invalid_gate_weight(self):
        with self.assertRaises(ValueError):
            recover_target_probabilities(torch.tensor([.1]), torch.tensor([.2]),
                                         torch.tensor([0.]))

    def test_rejects_material_negative_count(self):
        with self.assertRaises(ValueError):
            recover_target_probabilities(torch.tensor([.2]), torch.tensor([.1]),
                                         torch.tensor([.5]))


if __name__ == "__main__":
    unittest.main()
