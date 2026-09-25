"""Synthetic metadata-only checks; never open or score course test text."""
from __future__ import annotations

import copy
import math
import unittest

from scripts.package_stage143_final_release import (
    FINAL, expected_files, read_json, sha, validate_freeze, validate_test,
)


class FinalReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.final = read_json(FINAL)
        self.freeze = {
            "status": "method_frozen_before_test",
            "method_frozen": True,
            "protocol": self.final["protocol"],
            "candidate": "stage143-openvino-order6",
            "qualification_sha256": sha(FINAL),
            "validation_bpb": self.final["validation_bpb"],
            "conservative_inference_asset_bytes": self.final["conservative_asset_bytes"],
            "dirty_entries": [],
            "frozen_at_utc": "2000-01-01T00:00:00+00:00",
            "source_commit": "0" * 40,
            "inference_files": expected_files(self.final),
        }
        # The value 2.0 is synthetic metadata for gate tests, not an MP1 score.
        self.test_record = {
            "protocol": self.final["protocol"],
            "split": "test", "device": "cpu", "precision": "fp32",
            "targets": 428405, "utf8_bytes": 1292013,
            "checkpoint_sha256": self.final["checkpoint_sha256"],
            "implementation_sha256": self.final["source_hashes"]["student_stage143_openvino_singlepass.py"],
            "evaluator_sha256": self.final["source_hashes"]["evaluate.py"],
            "tokenizer_sha256": self.final["source_hashes"]["data/tokenizer.json"],
            "bpb": 2.0,
            "nll_nats": 2.0 * math.log(2) * 1292013,
        }

    def test_valid_synthetic_metadata(self) -> None:
        self.assertEqual(len(validate_freeze(self.freeze, self.final, sha(FINAL))), 18)
        validate_test(self.test_record, self.final)

    def test_unfrozen_or_changed_assets_rejected(self) -> None:
        for key, value in (("status", "eligible_for_freeze_only"),
                           ("method_frozen", False),
                           ("dirty_entries", ["M code/evaluate.py"])):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.freeze)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_freeze(bad, self.final, sha(FINAL))
        bad = copy.deepcopy(self.freeze)
        bad["inference_files"]["evaluate.py"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_freeze(bad, self.final, sha(FINAL))

    def test_wrong_test_identity_or_arithmetic_rejected(self) -> None:
        for key, value in (("split", "validation"), ("device", "cuda:0"),
                           ("precision", "bf16"), ("targets", 1),
                           ("utf8_bytes", 1), ("checkpoint_sha256", "0" * 64),
                           ("evaluator_sha256", "0" * 64), ("bpb", 1.0)):
            with self.subTest(key=key):
                bad = dict(self.test_record)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_test(bad, self.final)


if __name__ == "__main__":
    unittest.main()
