import math
import unittest

import torch

import student_hybrid_conv_output_bias
import student_ngram
import student_ngram_hybrid_conv_bias_collapsed


class OutputBiasCollapsedTests(unittest.TestCase):
    @staticmethod
    def config():
        neural = dict(
            vocab=2048, context=256, width=32, heads=4, depth=4,
            position="rope", norm="rmsnorm", activation="swiglu",
            mlp_ratio=2.5, bias=False, dropout=.1,
            output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
            conv_layers=[2, 4], conv_kernel=7, output_bias=True,
        )
        return dict(
            vocab=2048, context=256, kind="hybrid", max_order=2,
            min_count=2, discount=[.75], order_shapes=[[0, 0]],
            mixture_weight=.125, neural_config=neural,
        )

    def test_matches_direct_mixture_and_normalizes(self):
        config = self.config()
        model = student_ngram_hybrid_conv_bias_collapsed.build_model(config).eval()
        neural = student_hybrid_conv_output_bias.build_model(
            config["neural_config"]
        ).eval()
        count = student_ngram.NgramLM(config).eval()
        with torch.no_grad():
            model.neural.output_bias.normal_(std=.1)
            model.ngram.unigram.copy_(torch.rand(2048).add(.1))
            model.ngram.unigram.div_(model.ngram.unigram.sum())
        neural.load_state_dict(model.neural.state_dict())
        count.load_state_dict(model.ngram.state_dict())
        ids = torch.randint(2048, (2, 24))
        with torch.inference_mode():
            actual = model(ids)
            expected = torch.logaddexp(
                neural(ids) + math.log1p(-config["mixture_weight"]),
                count(ids) + math.log(config["mixture_weight"]),
            )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
            atol=2e-6, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
