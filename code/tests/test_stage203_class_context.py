"""Small normalization and causality tests for Stage203 class correction."""
import unittest

import numpy as np
import torch

from scripts.diagnose_stage203_class_context import (
    CLASSES, class_transition_tables, corrected_log_probs,
    next_class_probabilities,
)


class ClassContextTests(unittest.TestCase):
    def test_smoothed_rows_normalize(self):
        train = np.arange(1000, dtype=np.int64) % CLASSES
        bigram, trigram, evidence = class_transition_tables(train)
        np.testing.assert_allclose(bigram.sum(-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(trigram.sum(-1), 1.0, atol=1e-12)
        self.assertEqual(evidence["train_bigram_observations"], 999)

    def test_full_output_normalized_and_causal_class_rows(self):
        labels = torch.arange(2048) % CLASSES
        ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
        bigram = torch.full((CLASSES, CLASSES), 1 / CLASSES, dtype=torch.float64)
        trigram = torch.full((CLASSES, CLASSES, CLASSES), 1 / CLASSES,
                             dtype=torch.float64)
        q = next_class_probabilities(ids, labels, bigram, trigram)
        raw = torch.randn((2, 3, 2048))
        logp = raw.log_softmax(-1)
        corrected, mass = corrected_log_probs(logp, labels, q)
        torch.testing.assert_close(corrected.logsumexp(-1),
                                   torch.zeros((2, 3), dtype=torch.float64),
                                   atol=1e-10, rtol=0)
        self.assertTrue(torch.all(mass > 0))
        changed = ids.clone(); changed[0, 2] += 1
        changed_q = next_class_probabilities(changed, labels, bigram, trigram)
        torch.testing.assert_close(changed_q[0, :2], q[0, :2], atol=0, rtol=0)
        torch.testing.assert_close(changed_q[1], q[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
