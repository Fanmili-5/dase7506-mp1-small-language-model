import unittest

import numpy as np
import torch

from scripts.build_kneser_ney import effective_counts, fit_kneser_ney, modified_discounts


class KneserNeyTests(unittest.TestCase):
    def test_lower_orders_use_distinct_left_continuations(self):
        ids = np.array([0, 1, 2, 0, 1, 2, 3, 1, 2, 0], dtype=np.int64)
        grams, counts = effective_counts(ids, 2, 3)
        observed = dict(zip(grams.tolist(), counts.tolist()))
        # Bigram (1,2) has two distinct left predecessors: 0 and 3. Its raw
        # frequency is three, proving that this is continuation, not raw count.
        self.assertEqual(observed[1 * 2048 + 2], 2)

    def test_discounts_are_bounded_by_count_buckets(self):
        discounts, evidence = modified_discounts(np.array([1] * 20 + [2] * 8 + [3] * 4 + [4] * 2))
        self.assertFalse(evidence["fallback"])
        self.assertTrue(0 < discounts[0] < 1)
        self.assertTrue(0 < discounts[1] < 2)
        self.assertTrue(0 < discounts[2] < 3)

    def test_complete_positive_normalized_causal_distribution(self):
        ids = np.array(([0, 1, 2, 0, 1, 3, 0, 1, 2, 4] * 4), dtype=np.int64)
        model, config, summary = fit_kneser_ney(ids, max_order=4, min_count=1)
        self.assertEqual(config["estimator"], "pruned_interpolated_modified_kneser_ney")
        self.assertEqual(len(summary), 3)
        model.eval()
        x = torch.tensor([[0, 1, 2, 0, 1, 3], [4, 1, 2, 0, 1, 2]])
        with torch.inference_mode():
            distribution = model.distribution(x)
            changed = x.clone(); changed[:, -1] = 9
            changed_distribution = model.distribution(changed)
        self.assertTrue(torch.isfinite(distribution).all())
        self.assertTrue((distribution > 0).all())
        torch.testing.assert_close(distribution.sum(-1), torch.ones_like(distribution[..., 0]), atol=2e-6, rtol=0)
        torch.testing.assert_close(distribution[:, :-1], changed_distribution[:, :-1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
