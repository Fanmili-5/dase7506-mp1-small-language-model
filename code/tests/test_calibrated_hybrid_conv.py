import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts.build_ngram import fit_tables
import student_hybrid_conv_output_bias
import student_ngram_hybrid_conv_bias_collapsed
import student_ngram_hybrid_conv_calibrated_collapsed
import student_ngram_hybrid_conv_calibrated_untied_collapsed


class CalibratedHybridConvTests(unittest.TestCase):
    def test_materialized_calibration_matches_direct_formula(self):
        torch.manual_seed(7)
        counts, count_config, _ = fit_tables(
            np.tile([4, 7, 4, 9, 7, 4, 6, 4, 2, 7], 8), min_count=3)
        neural_config = dict(
            vocab=2048, context=256, width=24, heads=4, depth=2,
            position="rope", rope_base=10000., norm="rmsnorm", norm_eps=1e-5,
            activation="swiglu", mlp_ratio=2., dropout=0., bias=False,
            scaled_residual_init=True, output_kind="prefix_copy", copy_dim=8,
            copy_gate_bias=-2., conv_layers=[2], conv_kernel=3, output_bias=True,
        )
        temperature, prior_weight, gate_shift, mixture = 1.10, .05, .25, .075
        config = dict(
            count_config, kind="hybrid_calibrated_conv",
            neural_config=neural_config, mixture_weight=mixture,
            vocabulary_temperature=temperature,
        )
        original = student_hybrid_conv_output_bias.build_model(neural_config).eval()
        candidate = student_ngram_hybrid_conv_calibrated_collapsed.build_model(
            config).eval()
        candidate.neural.load_state_dict(original.state_dict())
        candidate.ngram.load_state_dict(counts.state_dict())
        log_prior = torch.linspace(-9., -5., 2048)
        old_bias = original.output_bias.detach().clone()
        with torch.no_grad():
            candidate.neural.output_bias.copy_(
                old_bias / temperature + prior_weight * log_prior)
            candidate.neural.copy_gate.bias.add_(gate_shift)
        ids = torch.tensor([[4, 7, 4, 9, 7], [3, 8, 3, 1, 6]])
        with torch.inference_mode():
            hidden = original.features(ids).float()
            vocabulary = F.softmax(
                (original.head(hidden) + old_bias) / temperature
                + prior_weight * log_prior,
                dim=-1,
            )
            copy = original.copy_distribution(hidden, ids)
            gate = original.copy_gate(hidden) + gate_shift
            expected = vocabulary * (torch.sigmoid(-gate) * (1 - mixture))
            expected.addcmul_(copy, torch.sigmoid(gate) * (1 - mixture))
            candidate.ngram.add_into(expected, ids, mixture)
            expected.div_(expected.sum(-1, keepdim=True))
            actual = candidate.predict_log_probs(ids)
        torch.testing.assert_close(actual.exp(), expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
            atol=2e-6, rtol=0)

        optimized_config = dict(config, kind="hybrid_calibrated_untied_conv")
        optimized = (
            student_ngram_hybrid_conv_calibrated_untied_collapsed.build_model(
                optimized_config).eval())
        optimized.load_state_dict(candidate.state_dict(), strict=True)
        with torch.no_grad():
            optimized.neural.head.weight.div_(temperature)
        with torch.inference_mode():
            optimized_output = optimized.predict_log_probs(ids)
        torch.testing.assert_close(
            optimized_output.exp(), actual.exp(), atol=2e-7, rtol=2e-6)

        folded_config = dict(config, kind="hybrid")
        folded_config.pop("vocabulary_temperature")
        folded = student_ngram_hybrid_conv_bias_collapsed.build_model(
            folded_config).eval()
        folded.load_state_dict(candidate.state_dict(), strict=True)
        with torch.no_grad():
            folded.neural.norm.weight.div_(temperature)
            if getattr(folded.neural.norm, "bias", None) is not None:
                folded.neural.norm.bias.div_(temperature)
            folded.neural.copy_query.weight.mul_(temperature)
            folded.neural.copy_key.weight.mul_(temperature)
            folded.neural.copy_gate.weight.mul_(temperature)
        with torch.inference_mode():
            folded_output = folded.predict_log_probs(ids)
        torch.testing.assert_close(
            folded_output.exp(), actual.exp(), atol=2e-7, rtol=2e-6)

    def test_invalid_temperature_is_rejected(self):
        counts, count_config, _ = fit_tables(np.tile([2, 4, 2, 5], 20), min_count=3)
        del counts
        config = dict(
            count_config, kind="hybrid_calibrated_conv", mixture_weight=.1,
            vocabulary_temperature=float("nan"),
            neural_config=dict(
                vocab=2048, context=256, width=24, heads=4, depth=2,
                position="rope", rope_base=10000., norm="rmsnorm", norm_eps=1e-5,
                activation="swiglu", mlp_ratio=2., dropout=0., bias=False,
                scaled_residual_init=True, output_kind="prefix_copy", copy_dim=8,
                copy_gate_bias=-2., conv_layers=[2], conv_kernel=3,
                output_bias=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            student_ngram_hybrid_conv_calibrated_collapsed.build_model(config)


if __name__ == "__main__":
    unittest.main()
