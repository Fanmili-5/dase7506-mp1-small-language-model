"""Causality, row reset and zero-scale parity for the bigram architecture."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop
import student_stage204_hashed_bigram_rdrop


ROOT = Path(__file__).resolve().parents[1]


class HashedBigramTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        self.candidate = json.loads((ROOT / "configs/stage204_hashed_bigram_rdrop.json").read_text())

    def test_zero_scale_exact_base_and_nonzero_causal(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        baseline = student_hybrid_conv_rdrop.build_model(self.base).eval()
        torch.manual_seed(17)
        zero = dict(self.candidate, bigram_scale=0.0)
        matched = student_stage204_hashed_bigram_rdrop.build_model(zero).eval()
        ids = torch.randint(0, 2048, (2, 16))
        with torch.inference_mode():
            torch.testing.assert_close(matched.predict_log_probs(ids),
                                       baseline.predict_log_probs(ids), atol=0, rtol=0)
            matched.bigram_scale = 0.5
            original = matched.predict_log_probs(ids)
            changed = ids.clone()
            changed[0, 12] = (changed[0, 12] + 1) % 2048
            changed_logp = matched.predict_log_probs(changed)
            torch.testing.assert_close(original[0, :12], changed_logp[0, :12],
                                       atol=1e-5, rtol=0)
            torch.testing.assert_close(original[1], changed_logp[1],
                                       atol=1e-5, rtol=0)
            self.assertLess(float(original.logsumexp(-1).abs().max()), 1e-5)
            self.assertTrue(torch.equal(matched.bigram_ids(ids)[:, 0],
                                        torch.zeros(2, dtype=torch.long)))

    def test_reduced_hash_is_exactly_the_predeclared_hash(self):
        torch.manual_seed(204017)
        ids = torch.randint(0, 2048, (128, 256), dtype=torch.long)
        model = student_stage204_hashed_bigram_rdrop.build_model(self.candidate)
        old = ((ids[:, :-1] * 1_315_423_911
                + ids[:, 1:] * 2_654_435_761) % 8192) + 1
        torch.testing.assert_close(model.bigram_ids(ids)[:, 1:], old,
                                   atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
