import unittest

import torch

from student_stage136_successor_neural import (build_model,
                                               calibrated_successor_log_probs)


class SuccessorNeuralTests(unittest.TestCase):
    @staticmethod
    def config():
        return dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                    position="rope", norm="rmsnorm", activation="swiglu",
                    mlp_ratio=2., bias=False, dropout=0.,
                    output_kind="prefix_copy", copy_dim=8,
                    conv_layers=[2], conv_kernel=3, output_bias=True)

    def test_causal_normalized_with_empty_first_copy(self):
        torch.manual_seed(136)
        model = build_model(self.config()).eval()
        ids = torch.randint(2048, (2, 9))
        altered = ids.clone()
        altered[:, 6:] = (altered[:, 6:] + 1) % 2048
        prior = torch.full((2048,), -torch.log(torch.tensor(2048.)).item())
        with torch.no_grad():
            logp = calibrated_successor_log_probs(model, ids, prior)
            changed = calibrated_successor_log_probs(model, altered, prior)
            short = calibrated_successor_log_probs(model, ids[:, :6], prior)
            copy = model.copy_distribution(model.features(ids), ids)
        self.assertTrue(torch.isfinite(logp).all())
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, 9),
                                   atol=3e-6, rtol=0)
        torch.testing.assert_close(logp[:, :6], changed[:, :6], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logp[:, :6], short, atol=1e-5, rtol=1e-5)
        self.assertEqual(float(copy[:, 0].sum()), 0.)
        torch.testing.assert_close(copy[:, 1:].sum(-1), torch.ones(2, 8),
                                   atol=1e-6, rtol=0)

    def test_copy_head_gradients_without_backbone_gradients(self):
        torch.manual_seed(137)
        model = build_model(self.config()).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.copy_query, model.copy_key, model.copy_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        ids = torch.randint(2048, (2, 8))
        prior = torch.full((2048,), -torch.log(torch.tensor(2048.)).item())
        logp = calibrated_successor_log_probs(model, ids, prior)
        loss = -logp[:, 1:].gather(-1, ids[:, 1:, None]).mean()
        loss.backward()
        for module in (model.copy_query, model.copy_key, model.copy_gate):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNone(model.blocks[0].mlp.input.weight.grad)


if __name__ == "__main__":
    unittest.main()
