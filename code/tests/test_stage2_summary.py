"""Synthetic arithmetic checks; no benchmark text or trained weights are used."""
import unittest

from scripts.summarize_stage2 import paired_summary


class Stage2SummaryTests(unittest.TestCase):
    def rows(self):
        return [dict(method=method, seed=seed, bpb=base + offset)
                for method, offset in [("baseline", 0), ("rope", -0.1), ("modern", -0.2)]
                for seed, base in [(17, 2.0), (23, 2.1), (42, 2.2)]]

    def test_all_seeds_are_included_and_paired(self):
        summary = paired_summary(list(reversed(self.rows())))
        self.assertAlmostEqual(summary["modern"]["mean_bpb"], 1.9)
        self.assertAlmostEqual(summary["baseline"]["sample_std_bpb"], 0.1)
        self.assertEqual(summary["modern"]["seeds"], [17, 23, 42])
        self.assertEqual(summary["modern"]["seeds_better_than_baseline"], 3)
        self.assertAlmostEqual(summary["modern"]["mean_paired_delta"], -0.2)

    def test_missing_seed_fails_instead_of_selecting_available_seeds(self):
        with self.assertRaises(KeyError):
            paired_summary(self.rows()[:-1])

    def test_duplicate_seed_fails(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            paired_summary(rows + [rows[0]])
