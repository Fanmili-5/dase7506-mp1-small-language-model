"""Architecture identity and causal controls for the Stage202 ordering."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_hybrid_conv_structured
from scripts.preflight_stage202_ordered_hybrid import check_config

ROOT = Path(__file__).resolve().parents[1]


class OrderedHybridTest(unittest.TestCase):
    def test_fixed_order_equal_size_and_exact_export(self):
        config, control = check_config()
        torch.manual_seed(17)
        candidate = student_hybrid_conv_rdrop.build_model(config)
        torch.manual_seed(17)
        reference = student_hybrid_conv_rdrop.build_model(control)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()),
                         sum(p.numel() for p in reference.parameters()))
        self.assertEqual(candidate.conv_layers, (1, 2, 3, 4))
        self.assertEqual(reference.conv_layers, (2, 4, 6, 8))
        deployed = student_hybrid_conv_structured.build_model(
            student_hybrid_conv_rdrop.inference_config(config))
        deployed.load_state_dict(
            student_hybrid_conv_rdrop.inference_state(candidate.state_dict()),
            strict=True)
        candidate.eval(); deployed.eval()
        ids = torch.randint(2048, (2, 16))
        with torch.inference_mode():
            logp = candidate.predict_log_probs(ids)
            torch.testing.assert_close(deployed(ids), logp, atol=0, rtol=0)
            torch.testing.assert_close(logp.logsumexp(-1),
                                       torch.zeros_like(logp[..., 0]), atol=2e-6, rtol=0)
            changed = ids.clone(); changed[0, -1] = (changed[0, -1] + 1) % 2048
            torch.testing.assert_close(candidate(changed)[0, :-1],
                                       logp[0, :-1], atol=1e-5, rtol=0)
            changed = ids.clone(); changed[1] = (changed[1] + 1) % 2048
            torch.testing.assert_close(candidate(changed)[0], logp[0],
                                       atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
