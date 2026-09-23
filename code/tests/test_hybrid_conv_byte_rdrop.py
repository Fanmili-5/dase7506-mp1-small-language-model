import unittest

import torch

import student_hybrid_conv_byte_rdrop
import student_hybrid_conv_rdrop
from scripts.export_stage65_hybrid_conv_byte_rdrop import materialize_model


class HybridConvByteRDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def config():
        return dict(
            vocab=2048, context=256, width=32, heads=4, depth=4,
            position="rope", norm="rmsnorm", activation="swiglu",
            mlp_ratio=2.5, bias=False, dropout=.1,
            output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
            embedding_row_dropout=.1, ffn_hidden_dropout=0.,
            deep_supervision_layers=[2, 3], deep_supervision_weight=.2,
            future_prediction_offsets=[2, 3], future_prediction_weight=.2,
            conv_layers=[2, 4], conv_kernel=7, rdrop_alpha=.5,
            byte_features="bag_first_last",
        )

    def test_zero_initialization_matches_parent_and_rng(self):
        config = self.config()
        torch.manual_seed(17)
        composed = student_hybrid_conv_byte_rdrop.build_model(config).eval()
        composed_rng = torch.get_rng_state()
        torch.manual_seed(17)
        parent = student_hybrid_conv_rdrop.build_model(config).eval()
        torch.testing.assert_close(composed_rng, torch.get_rng_state(), atol=0, rtol=0)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(composed.state_dict()[name], value, atol=0, rtol=0)
        ids = torch.randint(2048, (2, 24))
        with torch.inference_mode():
            torch.testing.assert_close(composed(ids), parent(ids), atol=0, rtol=0)

    def test_projection_gradient_and_exact_materialization(self):
        config = self.config()
        model = student_hybrid_conv_byte_rdrop.build_model(config).train()
        extended = torch.randint(2048, (2, 12))
        ids = extended[:, :9]
        targets = extended[:, 1:10]
        future = torch.stack((extended[:, 2:11], extended[:, 3:12]))
        loss, parts = model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(parts["symmetric_kl"]), -1e-6)
        loss.backward()
        self.assertIsNotNone(model.byte_projection.grad)
        self.assertTrue(torch.isfinite(model.byte_projection.grad).all())
        self.assertGreater(model.byte_projection.grad.abs().sum().item(), 0)
        model.eval()
        source, deployed, deployed_config = materialize_model(config, model.state_dict())
        self.assertNotIn("byte_features", deployed_config)
        self.assertNotIn("rdrop_alpha", deployed_config)
        self.assertNotIn("deep_supervision_layers", deployed_config)
        self.assertNotIn("future_prediction_offsets", deployed_config)
        source.eval(); deployed.eval()
        with torch.inference_mode():
            torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)

    def test_training_parameter_is_not_deployed(self):
        config = self.config()
        source = student_hybrid_conv_byte_rdrop.build_model(config)
        _, deployed, _ = materialize_model(config, source.state_dict())
        training_increment = sum(p.numel() for p in source.parameters()) - sum(
            p.numel() for p in student_hybrid_conv_rdrop.build_model(config).parameters()
        )
        self.assertEqual(training_increment, 768 * config["width"])
        self.assertNotIn("byte_projection", deployed.state_dict())


if __name__ == "__main__":
    unittest.main()
