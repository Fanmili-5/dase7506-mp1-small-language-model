import unittest

import torch

from student_stage135_successor_gate import add_successor_copy


class SuccessorCopyTransferTests(unittest.TestCase):
    def test_uniform_predecessor_alignment_and_normalization(self):
        ids = torch.tensor([[4, 7, 4, 7]])
        output = torch.zeros(1, 4, 8)
        scores = torch.zeros(1, 4, 4)
        scale = torch.tensor([[[0.], [1.], [1.], [1.]]])
        actual = add_successor_copy(output, ids, scores, scale)
        expected = torch.zeros_like(actual)
        expected[0, 1, 7] = 1.
        expected[0, 2, 7] = .5
        expected[0, 2, 4] = .5
        expected[0, 3, 7] = 2 / 3
        expected[0, 3, 4] = 1 / 3
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    def test_future_tokens_and_scores_cannot_change_prefix(self):
        ids = torch.tensor([[4, 7, 4, 7, 6]])
        changed = ids.clone()
        changed[0, 4] = 5
        scores = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(135))
        changed_scores = scores.clone()
        changed_scores[0, :4, 4] = 99.
        scale = torch.tensor([[[0.], [0.7], [0.7], [0.7], [0.7]]])
        a = add_successor_copy(torch.zeros(1, 5, 8), ids, scores, scale)
        b = add_successor_copy(torch.zeros(1, 5, 8), changed, changed_scores, scale)
        torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(a.sum(-1), scale.squeeze(-1), atol=1e-7, rtol=1e-7)

    def test_single_token_has_no_successor(self):
        output = torch.zeros(2, 1, 8)
        actual = add_successor_copy(output, torch.tensor([[1], [2]]),
                                    torch.zeros(2, 1, 1), torch.zeros(2, 1, 1))
        torch.testing.assert_close(actual, output)


if __name__ == "__main__":
    unittest.main()
