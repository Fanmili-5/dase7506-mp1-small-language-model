import math
import unittest

import torch
from torch.nn import functional as F

import student_calibrated_collapsed
import student_ngram


class CalibratedCollapsedTests(unittest.TestCase):
    def pair(self):
        base = dict(
            kind="hybrid", mixture_weight=.1125, vocab=2048, context=256,
            order_shapes=[(3, 5), (2, 3)],
            neural_config=dict(vocab=2048, context=256, width=24, heads=4, depth=2,
                               position="rope", norm="rmsnorm", activation="swiglu",
                               mlp_ratio=2, bias=False, dropout=.1,
                               output_kind="prefix_copy", copy_dim=8),
        )
        torch.manual_seed(43)
        old = student_ngram.build_model(base).eval()
        with torch.no_grad():
            old.ngram.unigram.copy_(torch.rand(2048).add(.1).div(torch.rand(2048).add(.1).sum()))
            # Use empty higher-order tables in this synthetic equality test.
            for table in old.ngram.tables:
                table.keys.zero_(); table.offsets.zero_(); table.values.zero_()
                table.mass.zero_(); table.backoff.fill_(1)
            old.ngram.unigram.div_(old.ngram.unigram.sum())
        prior = torch.linspace(-9, -5, 2048)
        config = dict(base, kind="hybrid_calibrated", vocabulary_temperature=1.075,
                      unigram_prior_weight=.05, copy_gate_shift=.25,
                      calibration_log_prior=prior.tolist())
        new = student_calibrated_collapsed.build_model(config).eval()
        new.neural.load_state_dict(old.neural.state_dict())
        new.ngram.load_state_dict(old.ngram.state_dict())
        return old, new, prior

    def test_matches_direct_probability_formula_and_normalizes(self):
        old, new, prior = self.pair()
        ids = torch.tensor([[4, 7, 4, 9, 2, 4], [3, 8, 3, 1, 6, 3]])
        with torch.inference_mode():
            hidden = old.neural.features(ids).float()
            vocabulary = F.softmax(old.neural.head(hidden) / 1.075 + .05 * prior, dim=-1)
            copy = old.neural.copy_distribution(hidden, ids)
            gate = old.neural.copy_gate(hidden) + .25
            expected = ((1 - .1125) * (torch.sigmoid(-gate) * vocabulary
                        + torch.sigmoid(gate) * copy) + .1125 * old.ngram.distribution(ids))
            expected.div_(expected.sum(-1, keepdim=True))
            actual = new.predict_log_probs(ids).exp()
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=2e-6)
        torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]),
                                   atol=2e-6, rtol=0)

    def test_training_path_has_finite_neural_gradients(self):
        _, model, _ = self.pair()
        model.train()
        ids = torch.tensor([[4, 7, 4, 9], [3, 8, 3, 1]])
        targets = torch.tensor([[7, 4, 9, 2], [8, 3, 1, 6]])
        loss = F.nll_loss(model(ids).flatten(0, 1), targets.flatten())
        loss.backward()
        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(any(parameter.grad is not None for parameter in model.neural.parameters()))


if __name__ == "__main__":
    unittest.main()
