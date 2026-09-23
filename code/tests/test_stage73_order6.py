import unittest

import numpy as np
import torch

from scripts.build_kneser_ney import fit_kneser_ney
from scripts.build_stage73_order6 import extend_model
from train_experiment import checkpoint_payload


class Order6ExtensionTests(unittest.TestCase):
    def test_extension_preserves_lower_tables_and_normalizes(self):
        ids = np.array(([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 6] * 5),
                       dtype=np.int64)
        base_model, config, _ = fit_kneser_ney(ids, max_order=5, min_count=1)
        base = checkpoint_payload(base_model, "student_ngram", config, 0,
                                  len(ids) - 1)
        extended, extended_config, summary = extend_model(base, ids, min_count=1)
        self.assertEqual(extended_config["max_order"], 6)
        self.assertEqual(len(extended.tables), 5)
        self.assertGreater(summary["retained_edges"], 0)
        for name, value in base_model.state_dict().items():
            torch.testing.assert_close(
                extended.state_dict()[name], value, atol=0, rtol=0
            )
        x = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 6]])
        with torch.inference_mode():
            probability = extended.distribution(x)
        self.assertTrue((probability > 0).all())
        torch.testing.assert_close(
            probability.sum(-1), torch.ones_like(probability[..., 0]),
            atol=2e-6, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
