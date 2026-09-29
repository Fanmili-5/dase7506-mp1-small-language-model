"""Arithmetic and fixed-cell checks for the Stage218 diagnostic."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage218_frozen_ceiling import CELLS, RAW_BYTES, mix_bpb


class Stage218FrozenCeilingTests(unittest.TestCase):
    def test_cells_are_fixed_convex_combinations(self):
        self.assertEqual(tuple(CELLS), ("A", "B", "C", "D", "E"))
        for weights in CELLS.values():
            self.assertTrue(all(weight >= 0 for weight in weights))
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_probability_mixture_is_normalized_arithmetic(self):
        arrays = tuple(np.full(5, math.log(value), dtype=np.float64)
                       for value in (0.2, 0.4, 0.6))
        measured = mix_bpb(arrays, (0.5, 0.25, 0.25))
        expected = -5 * math.log(0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.6) / math.log(2) / RAW_BYTES
        self.assertAlmostEqual(measured, expected)


if __name__ == "__main__":
    unittest.main()
