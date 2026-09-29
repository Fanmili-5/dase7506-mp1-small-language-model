"""Synthetic count-table tests; no validation or test corpus is scored."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.build_stage188_witten_bell import (
    raw_order6_counts, witten_bell_same_support,
)


class WittenBellTests(unittest.TestCase):
    def test_same_support_rows_normalize(self) -> None:
        contexts = np.array([3, 3, 3, 8, 8], dtype=np.int64)
        values = np.array([4, 7, 9, 1, 2], dtype=np.int64)
        counts = np.array([2, 4, 1, 3, 2], dtype=np.int64)
        expected = {
            "keys": torch.tensor([3, 8]),
            "offsets": torch.tensor([0, 2, 4]),
            "values": torch.tensor([4, 7, 1, 2], dtype=torch.int32),
            "mass": torch.zeros(4),
            "backoff": torch.zeros(2),
        }
        mass, backoff, summary = witten_bell_same_support(
            contexts, values, counts, expected)
        self.assertTrue(torch.allclose(mass, torch.tensor([.25, .5, 3 / 7, 2 / 7])))
        self.assertTrue(torch.allclose(backoff, torch.tensor([.25, 2 / 7])))
        self.assertLess(summary["max_context_normalization_error_float32"], 2e-6)

    def test_changed_support_rejected(self) -> None:
        expected = {
            "keys": torch.tensor([3]),
            "offsets": torch.tensor([0, 1]),
            "values": torch.tensor([5], dtype=torch.int32),
            "mass": torch.zeros(1),
            "backoff": torch.zeros(1),
        }
        with self.assertRaisesRegex(ValueError, "CSR support changed"):
            witten_bell_same_support(np.array([3]), np.array([4]),
                                     np.array([2]), expected)

    def test_order6_counts_do_not_overflow_context_key(self) -> None:
        ids = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        contexts, values, counts = raw_order6_counts(ids)
        key = (((((1 * 2048 + 2) * 2048 + 3) * 2048 + 4) * 2048 + 5))
        index = np.flatnonzero((contexts == key) & (values == 6))
        self.assertEqual(len(index), 1)
        self.assertEqual(int(counts[index[0]]), 2)


if __name__ == "__main__":
    unittest.main()
