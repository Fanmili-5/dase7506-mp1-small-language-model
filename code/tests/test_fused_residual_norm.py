import unittest

import torch

import student_ngram_hybrid_conv_fused_copy_collapsed
import student_ngram_hybrid_conv_fused_residual_norm
from tests.test_fused_copy_collapsed import FusedCopyCollapsedTests


class FusedResidualNormTests(unittest.TestCase):
    def test_direct_log_matches_dense_normalization(self):
        torch.manual_seed(85)
        config = FusedCopyCollapsedTests.config()
        reference = student_ngram_hybrid_conv_fused_copy_collapsed.build_model(
            config).eval()
        candidate = student_ngram_hybrid_conv_fused_residual_norm.build_model(
            config).eval()
        with torch.no_grad():
            reference.neural.output_bias.normal_(std=.1)
            reference.ngram.unigram.copy_(torch.rand(2048).add(.1))
            reference.ngram.unigram.div_(reference.ngram.unigram.sum())
        candidate.load_state_dict(reference.state_dict(), strict=True)
        ids = torch.randint(2048, (2, 37))
        with torch.inference_mode():
            expected = reference.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=0)
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
            atol=1e-5, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
