"""Synthetic tests, not evidence of performance."""
from pathlib import Path
import math
import copy
import tempfile
import unittest
from unittest.mock import patch
import torch
from student_structured import build_model
from scripts import screen_stage19_mixture as scan
from scripts.profile_structured_cpu import head, validate_implementation
from scripts import audit_stage19_results as audit


class Stage19Tests(unittest.TestCase):
    def test_resource_audit_recomputes_rejects_and_detects_corruption(self):
        def section(digest, seconds):
            row = dict(protocol=audit.PROTOCOL, split="validation", precision="fp32",
                checkpoint_sha256=digest, targets=376599, utf8_bytes=1148007,
                bpb=1.5, nll_nats=1.5 * math.log(2) * 1148007, seconds=seconds,
                peak_rss_bytes=2_000_000_000,
                evaluator_sha256=audit.sha(audit.ROOT / "evaluate.py"),
                tokenizer_sha256=audit.sha(audit.ROOT / "data/tokenizer.json"))
            return dict(runs=[dict(row) for _ in range(3)], checkpoint_sha256=digest,
                median_seconds=seconds, max_peak_rss_bytes=2_000_000_000)
        report = dict(repeats=3, threads=4, device="cpu", split="validation", precision="fp32",
            baseline=section(audit.BASE_SHA, 10), candidate=section(audit.H_SHA, 60),
            candidate_to_baseline_time_ratio=6, within_five_x_time_limit=False,
            within_four_gib_peak_rss_limit=True)
        self.assertFalse(audit.resource(report, audit.H_SHA, 1.5)["cpu_pass"])
        broken = copy.deepcopy(report)
        broken["within_five_x_time_limit"] = True
        with self.assertRaises(AssertionError):
            audit.resource(broken, audit.H_SHA, 1.5)
        broken = copy.deepcopy(report)
        broken["candidate"]["runs"][0]["bpb"] = 1.4
        with self.assertRaises(ValueError):
            audit.resource(broken, audit.H_SHA, 1.5)

    def test_native_checkpoint_schema_without_embedded_source_hash(self):
        payload = {"implementation": "student_structured"}
        validate_implementation(payload, "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18")
        with self.assertRaises(ValueError):
            validate_implementation(payload, "changed")

    def test_fixed_inputs_and_small_grid(self):
        self.assertEqual(scan.WEIGHTS, (0.0, 0.05, 0.10))
        self.assertEqual(scan.REFERENCE_SHA, "2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            neural = path / "neural.pt"
            neural.write_bytes(b"wrong checkpoint")
            argv = ["scan", "--neural", str(neural), "--counts", str(neural), "--run-dir", str(path / "run")]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "Stage18 H"):
                scan.main()
            with patch("sys.argv", argv), patch.object(scan, "sha", return_value=scan.REFERENCE_SHA), self.assertRaisesRegex(ValueError, "counts"):
                scan.main()
            self.assertFalse((path / "run").exists())

    def test_profile_decomposition_matches_real_forward(self):
        config = dict(vocab=2048, context=256, width=16, heads=2, depth=1,
            output_kind="prefix_copy", copy_dim=4, position="rope", norm="rmsnorm",
            activation="swiglu", mlp_ratio=2, dropout=.1, bias=False)
        model = build_model(config).eval()
        ids = torch.randint(2048, (2, 256))
        with torch.inference_mode():
            result = head(model, model.features(ids), ids)
            torch.testing.assert_close(result, model(ids), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
