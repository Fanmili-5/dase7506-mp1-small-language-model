import math
import unittest

import torch

from student_ensemble import build_model


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        member = {
            "vocab": 2048,
            "width": 32,
            "heads": 4,
            "depth": 1,
            "context": 256,
            "position": "rope",
            "norm": "rmsnorm",
            "activation": "swiglu",
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "bias": False,
        }
        self.model = build_model({
            "vocab": 2048,
            "context": 256,
            "member_configs": [member, member],
            "weights": [0.8, 0.2],
        }).eval()

    def test_mixture_is_normalized_and_matches_definition(self):
        ids = torch.randint(0, 2048, (2, 9))
        with torch.no_grad():
            actual = self.model.predict_log_probs(ids)
            first = self.model.members[0].predict_log_probs(ids) + math.log(0.8)
            second = self.model.members[1].predict_log_probs(ids) + math.log(0.2)
        torch.testing.assert_close(actual, torch.logaddexp(first, second))
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros(2, 9), atol=1e-6, rtol=1e-6
        )

    def test_rejects_invalid_weights(self):
        config = dict(self.model.config)
        config["weights"] = [0.8, 0.3]
        with self.assertRaises(ValueError):
            build_model(config)


if __name__ == "__main__":
    unittest.main()
