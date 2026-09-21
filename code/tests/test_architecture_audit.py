"""Synthetic audit tests, not experiment results."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from common import PROTOCOL
from scripts.audit_architecture_screen import audit, check_score, check_resource


class ArchitectureAuditTests(unittest.TestCase):
    sources = {"evaluate.py": "evaluator", "data/tokenizer.json": "tokenizer", "model.py": "baseline-module"}

    def score(self, checkpoint="candidate", implementation="module"):
        return {"protocol": PROTOCOL, "split": "validation", "device": "cpu", "precision": "fp32",
                "targets": 376599, "utf8_bytes": 1148007, "bpb": 1.5,
                "nll_nats": 1.5 * math.log(2) * 1148007, "checkpoint_sha256": checkpoint,
                "implementation_sha256": implementation, "evaluator_sha256": "evaluator",
                "tokenizer_sha256": "tokenizer", "seconds": 2., "peak_rss_bytes": 1024}

    def resource(self):
        groups = {}
        for label, checkpoint, implementation in (("baseline", "baseline", "baseline-module"),
                                                   ("candidate", "candidate", "module")):
            groups[label] = {"checkpoint_sha256": checkpoint, "median_seconds": 2.,
                             "max_peak_rss_bytes": 1024,
                             "runs": [self.score(checkpoint, implementation) for _ in range(3)]}
        return {"split": "validation", "device": "cpu", "precision": "fp32", "threads": 4, "repeats": 3,
                **groups, "candidate_to_baseline_time_ratio": 1.,
                "within_five_x_time_limit": True, "within_four_gib_peak_rss_limit": True}

    def test_valid_raw_score_and_resource(self):
        self.assertEqual(check_score(self.score(), "candidate", "module", self.sources), 1.5)
        result = check_resource(self.resource(), "candidate", "baseline", "module", self.sources, 1.5)
        self.assertTrue(result["time_pass"] and result["ram_pass"])

    def test_legacy_cpu_probe_inherits_top_level_device(self):
        resource = self.resource()
        for label in ("baseline", "candidate"):
            for row in resource[label]["runs"]:
                del row["device"]
        self.assertTrue(check_resource(resource, "candidate", "baseline", "module", self.sources)["time_pass"])
        resource["candidate"]["runs"][0]["device"] = "cuda"
        with self.assertRaises(ValueError):
            check_resource(resource, "candidate", "baseline", "module", self.sources)

    def test_reject_protocol_hash_and_score_mismatches(self):
        for patch in ({"split": "test"}, {"precision": "bf16"}, {"targets": 100},
                      {"checkpoint_sha256": "different"}, {"implementation_sha256": "different"},
                      {"evaluator_sha256": "different"}, {"bpb": 0.9}, {"bpb": float("nan")}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                check_score(self.score() | patch, "candidate", "module", self.sources)

    def test_reject_forged_resource_aggregates_and_wrong_checkpoint_score(self):
        for patch in ({"repeats": 1}, {"threads": 8}, {"within_five_x_time_limit": False},
                      {"candidate_to_baseline_time_ratio": .1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                check_resource(self.resource() | patch, "candidate", "baseline", "module", self.sources)
        row = self.resource()
        row["candidate"]["runs"][1]["seconds"] = float("nan")
        with self.assertRaises(ValueError):
            check_resource(row, "candidate", "baseline", "module", self.sources)
        with self.assertRaises(ValueError):
            check_resource(self.resource(), "candidate", "baseline", "module", self.sources, 1.4)

    def test_reject_unfinished_or_missing_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for state in ({"status": "training"}, {"status": "completed", "candidates": []}):
                (directory / "screen.json").write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    audit(directory)


if __name__ == "__main__":
    unittest.main()
