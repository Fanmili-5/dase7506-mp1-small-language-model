"""Synthetic and archived checks for the same-host Stage170 resource auditor."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from scripts.audit_stage170_linux_resource import audit


ROOT = Path(__file__).resolve().parents[1]


class Stage170ResourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.windows = json.loads(
            (ROOT / "results/stage143-evidence/resources.json").read_text(encoding="utf-8-sig"))

    def test_archived_windows_record_passes_same_protocol_audit(self) -> None:
        result = audit(self.windows, threads=4)
        self.assertTrue(result["all_local_gates_pass"])
        self.assertAlmostEqual(result["candidate_to_baseline_time_ratio"],
                               3.6177016681343095)
        self.assertEqual(result["conservative_inference_asset_bytes"], 55_810_412)

    def test_checkpoint_identity_change_is_rejected(self) -> None:
        record = copy.deepcopy(self.windows)
        record["candidate"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate checkpoint"):
            audit(record, threads=4)

    def test_incomplete_validation_is_rejected(self) -> None:
        record = copy.deepcopy(self.windows)
        record["candidate"]["runs"][0]["targets"] -= 1
        with self.assertRaisesRegex(ValueError, "protocol/coverage"):
            audit(record, threads=4)


if __name__ == "__main__":
    unittest.main()
