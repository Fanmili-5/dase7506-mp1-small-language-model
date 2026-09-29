"""Synthetic metadata-only checks; never open or score course test text."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.package_stage143_final_release import (
    FINAL, expected_files, qualification_hashes, read_json, sha,
    validate_freeze, validate_test,
    validate_window_nll,
)
from scripts.run_stage143_frozen_test import preflight


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
        self.freeze_sha256 = "f" * 64
        # The value 2.0 is synthetic metadata for gate tests, not an MP1 score.
        self.test_record = {
            "protocol": self.final["protocol"],
            "split": "test", "device": "cpu", "precision": "fp32",
            "targets": 428405, "utf8_bytes": 1292013,
            "checkpoint_sha256": self.final["checkpoint_sha256"],
            "implementation_sha256": self.final["source_hashes"]["student_stage143_openvino_singlepass.py"],
            "evaluator_sha256": self.final["source_hashes"]["evaluate.py"],
            "tokenizer_sha256": self.final["source_hashes"]["data/tokenizer.json"],
            "freeze_record_sha256": self.freeze_sha256,
            "frozen_source_commit": self.freeze["source_commit"],
            "bpb": 2.0,
            "nll_nats": 2.0 * math.log(2) * 1292013,
        }

    def test_valid_synthetic_metadata(self) -> None:
        self.assertEqual(len(validate_freeze(self.freeze, self.final, sha(FINAL))), 18)
        validate_test(self.test_record, self.final, self.freeze_sha256,
                      self.freeze["source_commit"])

    def test_recorded_windows_qualification_matches_only_line_endings(self) -> None:
        recorded = read_json(FINAL.parent / "freeze-stage143-20260927.json")
        accepted = qualification_hashes(FINAL)
        self.assertIn(recorded["qualification_sha256"], accepted)
        self.assertEqual(len(validate_freeze(recorded, self.final, accepted)), 18)
        changed = copy.deepcopy(recorded)
        changed["qualification_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_freeze(changed, self.final, accepted)

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
                           ("evaluator_sha256", "0" * 64),
                           ("freeze_record_sha256", "0" * 64),
                           ("frozen_source_commit", "1" * 40), ("bpb", 1.0)):
            with self.subTest(key=key):
                bad = dict(self.test_record)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_test(bad, self.final, self.freeze_sha256,
                                  self.freeze["source_commit"])

    def test_scoring_refuses_without_freeze_record(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            with self.assertRaisesRegex(ValueError, "No method freeze record"):
                preflight(directory / "missing-freeze.json",
                          directory / "unscored-test.json")

    def test_gitless_preflight_binds_exact_freeze_hash(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            freeze_path = directory / "freeze.json"
            freeze_path.write_text(json.dumps(self.freeze), encoding="utf-8")
            output = directory / "unscored-test.json"
            with patch("scripts.run_stage143_frozen_test.REPO", directory):
                final, freeze, frozen_sha = preflight(freeze_path, output,
                                                       sha(freeze_path))
                self.assertEqual(final["validation_bpb"], self.final["validation_bpb"])
                self.assertEqual(freeze, self.freeze)
                self.assertEqual(frozen_sha, sha(freeze_path))
                with self.assertRaisesRegex(ValueError, "Portable freeze SHA-256"):
                    preflight(freeze_path, output, "0" * 64)
            self.assertFalse(output.exists())

    def test_window_sidecar_must_match_full_test_summary(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "synthetic.window-nll.npy"
            losses = np.full(1674, self.test_record["nll_nats"] / 1674,
                             dtype=np.float64)
            np.save(path, losses)
            self.assertEqual(validate_window_nll(self.test_record, path), 1674)
            losses[0] += 1
            np.save(path, losses)
            with self.assertRaisesRegex(ValueError, "Window-NLL sum"):
                validate_window_nll(self.test_record, path)
            np.save(path, losses[:-1])
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_window_nll(self.test_record, path)


if __name__ == "__main__":
    unittest.main()
