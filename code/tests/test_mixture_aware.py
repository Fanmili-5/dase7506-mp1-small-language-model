import unittest

import torch

import student_ngram
from student_mixture_aware import (build_target_edge_keys,
                                   count_target_probability, mixture_log_probs,
                                   mixture_target_log_probs, symmetric_kl)


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

    def test_sparse_target_query_matches_complete_count_distribution(self):
        generator = torch.Generator().manual_seed(71017)
        config = dict(context=256, vocab=2048, kind="ngram",
                      order_shapes=[[2, 3]])
        counts = student_ngram.NgramLM(config).eval()
        with torch.no_grad():
            counts.unigram.copy_(torch.rand(2048, generator=generator).add(.1))
            counts.unigram.div_(counts.unigram.sum())
            table = counts.tables[0]
            table.keys.copy_(torch.tensor([1, 2]))
            table.offsets.copy_(torch.tensor([0, 2, 3]))
            table.values.copy_(torch.tensor([3, 4, 5]))
            table.mass.copy_(torch.tensor([.1, .2, .3]))
            table.backoff.copy_(torch.tensor([.7, .6]))
        ids = torch.tensor([[1, 7, 2, 1], [8, 2, 9, 1]])
        targets = torch.tensor([[3, 6, 5, 4], [2, 5, 9, 7]])
        expected = counts.distribution(ids).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        edge_keys = build_target_edge_keys(counts)
        actual = count_target_probability(counts, ids, targets, edge_keys)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        neural = torch.randn(2, 4, 2048, generator=generator).log_softmax(-1)
        complete = mixture_log_probs(neural, counts(ids), .0625).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        sparse = mixture_target_log_probs(neural, actual, targets, .0625)
        # The two algebraically equivalent FP32 paths can differ by one or two
        # CPU/backend rounding units after logaddexp; keep this deterministic
        # and use the same 1e-6 scale as the normalization contract above.
        torch.testing.assert_close(sparse, complete, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
