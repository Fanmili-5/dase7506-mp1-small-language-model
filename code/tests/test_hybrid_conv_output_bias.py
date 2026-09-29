import unittest

import torch
from torch.nn import functional as F

import student_hybrid_conv_output_bias
import student_hybrid_conv_structured


class HybridConvOutputBiasTests(unittest.TestCase):
    @staticmethod
    def config():
        return dict(
            vocab=2048, context=256, width=32, heads=4, depth=4,
            position="rope", norm="rmsnorm", activation="swiglu",
            mlp_ratio=2.5, bias=False, dropout=.1,
            output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
            conv_layers=[2, 4], conv_kernel=7, output_bias=True,
        )

    def test_zero_bias_matches_parent_without_rng_change(self):
        config = self.config()
        torch.manual_seed(17)
        biased = student_hybrid_conv_output_bias.build_model(config).eval()
        biased_rng = torch.get_rng_state()
        torch.manual_seed(17)
        parent = student_hybrid_conv_structured.build_model(config).eval()
        torch.testing.assert_close(biased_rng, torch.get_rng_state(), atol=0, rtol=0)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(biased.state_dict()[name], value, atol=0, rtol=0)
        ids = torch.randint(2048, (2, 24))
        with torch.inference_mode():
            torch.testing.assert_close(biased(ids), parent(ids), atol=0, rtol=0)

    def test_only_bias_receives_gradient_and_output_is_normalized(self):
        model = student_hybrid_conv_output_bias.build_model(self.config()).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.output_bias.requires_grad_(True)
        ids = torch.randint(2048, (2, 16))
        targets = torch.randint(2048, ids.shape)
        logp = model(ids)
        loss = F.nll_loss(logp.flatten(0, 1), targets.flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(model.output_bias.grad).all())
        self.assertGreater(model.output_bias.grad.abs().sum().item(), 0)
        self.assertTrue(all(
            parameter.grad is None for name, parameter in model.named_parameters()
            if name != "output_bias"
        ))
        torch.testing.assert_close(
            logp.logsumexp(-1), torch.zeros_like(logp[..., 0]), atol=2e-6, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
