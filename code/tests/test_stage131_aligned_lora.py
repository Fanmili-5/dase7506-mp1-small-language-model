"""The aligned sampler covers full independent causal windows without drops."""
from __future__ import annotations

import unittest

import torch

from scripts.train_stage131_aligned_lora import AlignedWindowSampler


class AlignedWindowSamplerTests(unittest.TestCase):
    def test_seeded_full_epoch_and_target_shift(self):
        tokens = torch.arange(4 * 256 + 1)
        left = AlignedWindowSampler(tokens, batch=3, seed=129017)
        right = AlignedWindowSampler(tokens, batch=3, seed=129017)
        first_x, first_y = left.next()
        second_x, second_y = left.next()
        same_x, same_y = right.next()
        torch.testing.assert_close(first_x, same_x)
        torch.testing.assert_close(first_y, same_y)
        seen = torch.cat((first_x[:, 0], second_x[:1, 0])) // 256
        self.assertEqual(sorted(seen.tolist()), [0, 1, 2, 3])
        self.assertEqual(left.completed_epochs, 1)
        torch.testing.assert_close(first_y, first_x + 1)
        torch.testing.assert_close(second_y, second_x + 1)
        self.assertEqual(left.excluded_targets, 0)

    def test_excludes_only_last_partial_window(self):
        tokens = torch.arange(4 * 256 + 93)
        sampler = AlignedWindowSampler(tokens, batch=3, seed=129017)
        self.assertEqual(sampler.num_windows, 4)
        self.assertEqual(sampler.excluded_targets, 92)


if __name__ == "__main__":
    unittest.main()
