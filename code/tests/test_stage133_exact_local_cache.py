import unittest

import torch

from scripts.scan_stage43_exact_local_cache import local_successor_statistics
from student_stage133_exact_local_cache import add_exact_local_cache


class DeployableLocalCacheTests(unittest.TestCase):
    def test_normalization_and_target_probability_parity(self):
        ids = torch.tensor([[4, 7, 4, 7, 4, 7, 9],
                            [7, 4, 7, 4, 7, 4, 9]])
        targets = torch.tensor([[7, 4, 7, 4, 7, 9, 4],
                                [4, 7, 4, 7, 4, 9, 7]])
        generator = torch.Generator().manual_seed(133)
        base = torch.rand(2, 7, 16, generator=generator, dtype=torch.float64)
        base /= base.sum(-1, keepdim=True)
        actual = add_exact_local_cache(base.clone(), ids)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 7, dtype=torch.float64))
        for row in range(2):
            cache, order = local_successor_statistics(ids[row], targets[row])
            for position in range(7):
                token = int(targets[row, position])
                selected = float(base[row, position, token])
                if order[position] >= 2:
                    selected = 0.9 * selected + 0.1 * float(cache[position])
                self.assertAlmostEqual(float(actual[row, position, token]), selected)

    def test_future_and_other_window_do_not_affect_prefix(self):
        ids = torch.tensor([[4, 7, 4, 7, 4, 7],
                            [4, 7, 4, 7, 4, 7]])
        changed = ids.clone()
        changed[0, 5] = 8
        changed[1] = torch.tensor([1, 2, 3, 4, 5, 6])
        base = torch.full((2, 6, 12), 1 / 12, dtype=torch.float64)
        original = add_exact_local_cache(base.clone(), ids)
        altered = add_exact_local_cache(base.clone(), changed)
        torch.testing.assert_close(original[0, :5], altered[0, :5])
        torch.testing.assert_close(original[0], original[1])

    def test_invalid_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            add_exact_local_cache(torch.full((1, 2, 3), 1 / 3),
                                  torch.tensor([[0, 3]]))


if __name__ == "__main__":
    unittest.main()
