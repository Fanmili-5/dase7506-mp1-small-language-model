import unittest

import torch

from scripts.scan_stage43_exact_local_cache import local_successor_statistics


class ExactLocalCacheTests(unittest.TestCase):
    def test_successor_alignment_and_longest_order(self):
        ids = torch.tensor([4, 7, 4, 7])
        targets = torch.tensor([7, 4, 7, 4])
        probability, order = local_successor_statistics(ids, targets, 8)
        torch.testing.assert_close(probability, torch.tensor([0., 0., 1., 1.], dtype=torch.float64))
        torch.testing.assert_close(order, torch.tensor([0, 0, 1, 2]))

    def test_padding_is_ignored(self):
        ids = torch.tensor([4, 7, 4, 0, 0])
        targets = torch.tensor([7, 4, 9, -100, -100])
        probability, order = local_successor_statistics(ids, targets, 4)
        self.assertEqual(probability[3:].sum(), 0)
        self.assertEqual(order[3:].sum(), 0)


if __name__ == "__main__":
    unittest.main()
