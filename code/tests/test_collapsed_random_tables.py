"""Check sparse expansion with ragged count tables and missing contexts."""
import unittest
import numpy as np
import torch
from scripts.build_ngram import fit_tables
from student_ngram_collapsed import CollapsedNgramLM


class RandomCountTests(unittest.TestCase):
    def test_ragged_tables_match_original_recurrence(self):
        torch.set_num_threads(2)
        # Synthetic test RNG only: no training or validation-based seed selection.
        rng = np.random.default_rng(123)
        train = rng.integers(1, 9, size=1200)
        train[::7] = 4
        train[1::7] = 7
        train[2::7] = 4
        old, config, _ = fit_tables(train, min_count=2)
        new = CollapsedNgramLM(config)
        new.load_state_dict(old.state_dict())
        for length in (1, 3, 4, 17, 256):
            ids = torch.tensor(rng.integers(1, 12, size=(3,length)))
            # Include both training-derived suffixes and genuinely missing IDs.
            ids[0] = torch.tensor(train[:length])
            initial = torch.full((3,length,2048), 1/2048)
            for scale in (0., .1, .9):
                expected = initial + scale * old.distribution(ids)
                actual = new.add_into(initial.clone(), ids, scale)
                torch.testing.assert_close(actual, expected, rtol=3e-6, atol=6e-8)
                self.assertTrue(torch.isfinite(actual).all())
                repeated = new.add_into(initial.clone(), ids, scale)
                torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
