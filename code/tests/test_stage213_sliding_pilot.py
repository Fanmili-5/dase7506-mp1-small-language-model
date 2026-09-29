"""Stage213 matched-start and learning-rate-prefix invariants."""
import json
from pathlib import Path
import unittest

import torch

from scripts.preflight_stage212_sliding_local import matched_models
from scripts.run_stage213_sliding_pilot import (candidate_checkpoint_payload,
                                                  matched_learning_rate)
from train_experiment import learning_rate


ROOT = Path(__file__).resolve().parents[1]


class Stage213MatchedPilotTests(unittest.TestCase):
    def test_every_common_start_tensor_matches_stage54(self):
        config = json.loads(
            (ROOT / "configs/stage212_sliding_local_attention_rdrop.json")
            .read_text(encoding="utf-8"))
        control_config = json.loads(
            (ROOT / "configs/stage54_hybrid_conv_rdrop.json")
            .read_text(encoding="utf-8"))
        control, candidate, _, shared = matched_models(
            torch.device("cpu"), config, control_config)
        self.assertEqual(shared, 57)
        source, target = control.state_dict(), candidate.state_dict()
        for name in source.keys() & target.keys():
            torch.testing.assert_close(source[name], target[name], atol=0, rtol=0)
        self.assertEqual(candidate.local_attention_layers, (2, 4, 6, 8))
        payload = candidate_checkpoint_payload(
            candidate, "student_hybrid_conv_rdrop", config, 17, 8192)
        self.assertEqual(payload["implementation"],
                         "student_stage212_sliding_local_rdrop")
        self.assertEqual(payload["train_tokens"], 8192)

    def test_learning_rate_uses_control_schedule_prefix(self):
        for step in (0, 99, 100, 299, 1199, 2399):
            expected = learning_rate(step, 7200, 1e-3, 100, .1, "baseline")
            actual = matched_learning_rate(step, 2400, 1e-3, 100, .1, "baseline")
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
