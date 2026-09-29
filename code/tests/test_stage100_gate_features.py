import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts.analyze_stage50_confidence_gate import confidence_features
from scripts.build_kneser_ney import fit_kneser_ney
from scripts.fit_stage100_train_gate import confidence_features5


class Stage100GateFeatureTests(unittest.TestCase):
    def test_order5_features_match_prior_diagnostic(self):
        tokens = np.tile(np.array([2, 3, 5, 7, 11, 13, 17, 19]), 30)
        counts, _, _ = fit_kneser_ney(tokens, min_count=2)
        ids = torch.tensor(tokens[:32]).reshape(2, 16)
        torch.manual_seed(100)
        neural = F.log_softmax(torch.randn(2, 16, 2048), dim=-1)
        with torch.inference_mode():
            count_logp = counts.predict_log_probs(ids)
            expected = confidence_features(neural, count_logp, counts, ids)
            actual = confidence_features5(neural, count_logp, counts, ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
