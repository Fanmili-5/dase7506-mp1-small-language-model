import unittest

import torch

import student_multi_token
import student_untied_multi_token
from scripts.export_stage44_untied import materialize_model


class UntiedTests(unittest.TestCase):
    @staticmethod
    def config():
        return dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                    position="rope", norm="rmsnorm", activation="swiglu",
                    mlp_ratio=2.6666667, bias=False, dropout=.1,
                    output_kind="prefix_copy", copy_dim=8, copy_gate_bias=-2.,
                    embedding_row_dropout=.1, ffn_hidden_dropout=0.,
                    deep_supervision_layers=[1], deep_supervision_weight=.2,
                    future_prediction_offsets=[2, 3], future_prediction_weight=.2,
                    tie_embeddings=False)

    def test_head_is_independent_and_parameter_increment_exact(self):
        untied = student_untied_multi_token.build_model(self.config())
        tied = student_multi_token.build_model(self.config())
        self.assertNotEqual(untied.head.weight.data_ptr(), untied.token.weight.data_ptr())
        difference = sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters())
        self.assertEqual(difference, 2048 * 32)

    def test_training_gradients_and_export(self):
        model = student_untied_multi_token.build_model(self.config()).train()
        extended = torch.randint(2048, (2, 8))
        loss, _ = model.training_loss(
            extended[:, :5], extended[:, 1:6],
            torch.stack((extended[:, 2:7], extended[:, 3:8])))
        loss.backward()
        self.assertGreater(model.token.weight.grad.abs().sum(), 0)
        self.assertGreater(model.head.weight.grad.abs().sum(), 0)
        source, deployed, _ = materialize_model(self.config(), model.state_dict())
        source.eval(); deployed.eval(); ids = torch.randint(2048, (2, 16))
        with torch.no_grad():
            torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
