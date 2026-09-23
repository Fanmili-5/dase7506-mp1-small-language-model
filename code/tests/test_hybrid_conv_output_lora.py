import unittest

import torch
from torch.nn import functional as F

import student_hybrid_conv_output_bias
import student_hybrid_conv_output_lora
from scripts.export_stage69_output_lora import materialize_model


class HybridConvOutputLoRATests(unittest.TestCase):
    @staticmethod
    def config():
        return dict(
            vocab=2048, context=256, width=32, heads=4, depth=4,
            position="rope", norm="rmsnorm", activation="swiglu",
            mlp_ratio=2.5, bias=False, dropout=.1,
            output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
            conv_layers=[2, 4], conv_kernel=7, output_bias=True,
            output_lora_rank=4, output_lora_alpha=4.,
        )

    def test_zero_b_matches_parent_and_receives_gradient(self):
        config = self.config()
        torch.manual_seed(17)
        model = student_hybrid_conv_output_lora.build_model(config).eval()
        torch.manual_seed(17)
        parent = student_hybrid_conv_output_bias.build_model(config).eval()
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        ids = torch.randint(2048, (2, 16)); targets = torch.randint(2048, ids.shape)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), parent(ids), atol=0, rtol=0)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.output_lora_a.requires_grad_(True); model.output_lora_b.requires_grad_(True)
        loss = F.nll_loss(model(ids).flatten(0, 1), targets.flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(model.output_lora_b.grad).all())
        self.assertGreater(model.output_lora_b.grad.abs().sum().item(), 0)

    def test_materialized_untied_export_is_exact(self):
        config = self.config()
        model = student_hybrid_conv_output_lora.build_model(config).eval()
        with torch.no_grad():
            model.output_lora_b.normal_(std=.01)
        source, deployed, deployed_config = materialize_model(config, model.state_dict())
        self.assertTrue(deployed_config["untied_output"])
        self.assertNotIn("output_lora_rank", deployed_config)
        self.assertIsNot(deployed.token.weight, deployed.head.weight)
        ids = torch.randint(2048, (2, 24))
        source.eval(); deployed.eval()
        with torch.inference_mode():
            torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
