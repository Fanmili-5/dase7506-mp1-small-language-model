import json
from pathlib import Path
import unittest

import torch

import student_byte_composed
import student_multi_token
from scripts.export_stage42_byte_composed import materialize_model

ROOT = Path(__file__).resolve().parents[1]


class ByteComposedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def config():
        return dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                    position="rope", norm="rmsnorm", activation="swiglu",
                    mlp_ratio=2.6666667, bias=False, dropout=.1,
                    output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
                    embedding_row_dropout=.1, ffn_hidden_dropout=0.,
                    deep_supervision_layers=[1], deep_supervision_weight=.2,
                    future_prediction_offsets=[2, 3], future_prediction_weight=.2,
                    byte_features="bag_first_last")

    def test_reversible_byte_mapping(self):
        reverse = student_byte_composed.byte_unicode_reverse()
        self.assertEqual([reverse[c] for c in "ĠWhen"], [32, 87, 104, 101, 110])
        features = student_byte_composed.build_byte_features(2048)
        self.assertEqual(tuple(features.shape), (2048, 768))
        self.assertTrue(torch.isfinite(features).all())

    def test_zero_initialized_model_matches_parent_and_rng(self):
        config = self.config()
        torch.manual_seed(17)
        composed = student_byte_composed.build_model(config).eval()
        composed_rng = torch.get_rng_state()
        torch.manual_seed(17)
        parent = student_multi_token.build_model(config).eval()
        torch.testing.assert_close(composed_rng, torch.get_rng_state(), atol=0, rtol=0)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(composed.state_dict()[name], value, atol=0, rtol=0)
        ids = torch.randint(2048, (2, 32))
        with torch.no_grad():
            torch.testing.assert_close(composed(ids), parent(ids), atol=0, rtol=0)

    def test_composition_receives_finite_nonzero_gradients(self):
        model = student_byte_composed.build_model(self.config()).train()
        extended = torch.randint(2048, (2, 8))
        loss, _ = model.training_loss(
            extended[:, :5], extended[:, 1:6],
            torch.stack((extended[:, 2:7], extended[:, 3:8])))
        loss.backward()
        self.assertIsNotNone(model.byte_projection.grad)
        self.assertTrue(torch.isfinite(model.byte_projection.grad).all())
        self.assertGreater(model.byte_projection.grad.abs().sum().item(), 0)

    def test_full_parameter_increment_is_training_only(self):
        config = json.loads((ROOT / "configs/stage42_byte_composed.json").read_text())
        parent_config = json.loads((ROOT / "configs/stage26_multi_token.json").read_text())
        composed = student_byte_composed.build_model(config)
        parent = student_multi_token.build_model(parent_config)
        difference = sum(p.numel() for p in composed.parameters()) - sum(p.numel() for p in parent.parameters())
        self.assertEqual(difference, 768 * 256)

    def test_materialized_export_is_exact(self):
        model = student_byte_composed.build_model(self.config()).eval()
        with torch.no_grad():
            model.byte_projection.normal_(std=.01)
        source, deployed, config = materialize_model(self.config(), model.state_dict())
        self.assertNotIn("byte_features", config)
        ids = torch.randint(2048, (2, 32))
        source.eval(); deployed.eval()
        with torch.no_grad():
            torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
