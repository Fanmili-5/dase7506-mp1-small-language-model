import unittest

import torch

from student_mixture_aware import mixture_log_probs, symmetric_kl


class MixtureAwareTests(unittest.TestCase):
    def test_normalized_mixture_has_neural_gradient(self):
        neural_logits = torch.randn(2, 5, 11, requires_grad=True)
        count_logits = torch.randn(2, 5, 11)
        neural = neural_logits.log_softmax(-1)
        counts = count_logits.log_softmax(-1)
        mixed = mixture_log_probs(neural, counts, .0625)
        torch.testing.assert_close(
            mixed.logsumexp(-1), torch.zeros(2, 5), atol=1e-6, rtol=0
        )
        loss = -mixed[..., 3].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(neural_logits.grad).all())
        self.assertGreater(neural_logits.grad.abs().sum().item(), 0)

    def test_symmetric_kl_is_zero_only_for_equal_distributions(self):
        first = torch.randn(2, 5, 11).log_softmax(-1)
        self.assertAlmostEqual(float(symmetric_kl(first, first)), 0.0, places=7)
        second = torch.randn(2, 5, 11).log_softmax(-1)
        self.assertGreater(float(symmetric_kl(first, second)), 0)


if __name__ == "__main__":
    unittest.main()
