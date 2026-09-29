import unittest

import torch

import student_ngram_hybrid_conv_bias_collapsed
import student_ngram_hybrid_conv_fused_copy_collapsed


class FusedCopyCollapsedTests(unittest.TestCase):
    @staticmethod
    def config():
        neural = dict(
            vocab=2048, context=256, width=32, heads=4, depth=4,
            position="rope", norm="rmsnorm", activation="swiglu",
            mlp_ratio=2.5, bias=False, dropout=0., output_kind="prefix_copy",
            copy_dim=8, copy_gate_bias=-2., conv_layers=[2, 4],
            conv_kernel=7, output_bias=True,
        )
        return dict(
            vocab=2048, context=256, kind="hybrid", max_order=2,
            min_count=2, discount=[.75], order_shapes=[[0, 0]],
            mixture_weight=.075, neural_config=neural,
        )

    def test_matches_existing_collapsed_predictor(self):
        torch.manual_seed(84)
        config = self.config()
        reference = student_ngram_hybrid_conv_bias_collapsed.build_model(
            config).eval()
        candidate = student_ngram_hybrid_conv_fused_copy_collapsed.build_model(
            config).eval()
        with torch.no_grad():
            reference.neural.output_bias.normal_(std=.1)
            reference.ngram.unigram.copy_(torch.rand(2048).add(.1))
            reference.ngram.unigram.div_(reference.ngram.unigram.sum())
        candidate.load_state_dict(reference.state_dict(), strict=True)
        ids = torch.randint(2048, (2, 31))
        with torch.inference_mode():
            expected = reference.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
            atol=2e-6, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
