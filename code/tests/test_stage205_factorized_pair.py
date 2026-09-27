"""Causality, reset, normalization, and baseline parity for Stage205."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_stage205_factorized_pair_rdrop


ROOT = Path(__file__).resolve().parents[1]


class FactorizedPairTests(unittest.TestCase):
    def test_base_parity_and_causal_pair_residual(self):
        torch.set_num_threads(2)
        base_config = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        candidate_config = json.loads((ROOT / "configs/stage205_factorized_pair_rdrop.json").read_text())
        torch.manual_seed(17)
        baseline = student_hybrid_conv_rdrop.build_model(base_config).eval()
        torch.manual_seed(17)
        candidate = student_stage205_factorized_pair_rdrop.build_model(candidate_config).eval()
        ids = torch.randint(0, 2048, (2, 16))
        with torch.inference_mode():
            self.assertEqual(float(candidate.pair_residual(ids)[:, 0].abs().max()), 0)
            candidate.pair_scale = 0
            torch.testing.assert_close(candidate.predict_log_probs(ids),
                                       baseline.predict_log_probs(ids), atol=0, rtol=0)
            candidate.pair_scale = 0.75
            original = candidate.predict_log_probs(ids)
            changed = ids.clone()
            changed[0, 12] = (changed[0, 12] + 1) % 2048
            altered = candidate.predict_log_probs(changed)
            torch.testing.assert_close(original[0, :12], altered[0, :12],
                                       atol=1e-5, rtol=0)
            torch.testing.assert_close(original[1], altered[1], atol=1e-5, rtol=0)
            self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
            self.assertGreater(float(candidate.pair_residual(ids)[:, 1:].abs().max()), 0)


if __name__ == "__main__":
    unittest.main()
