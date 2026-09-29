import unittest

import torch

from scripts.scan_stage41_continuous_cache import successor_target_probability


class ContinuousCacheTests(unittest.TestCase):
    def test_uniform_alignment(self):
        hidden = torch.zeros(1, 4, 3)
        ids = torch.tensor([[4, 7, 4, 7]])
        targets = torch.tensor([[7, 4, 7, 4]])
        actual = successor_target_probability(hidden, ids, targets, 8.)
        # t=0 has no legal successor; later rows are uniform over j<t.
        expected = torch.tensor([[0., 0., 1/2, 1/3]])
        torch.testing.assert_close(actual, expected)

    def test_future_hidden_does_not_change_prefix(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 7, 5)
        changed = hidden.clone(); changed[:, 4:] += 100
        ids = torch.randint(20, (2, 7))
        targets = torch.randint(20, (2, 7))
        p = successor_target_probability(hidden, ids, targets, 4.)
        q = successor_target_probability(changed, ids, targets, 4.)
        torch.testing.assert_close(p[:, :4], q[:, :4], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
