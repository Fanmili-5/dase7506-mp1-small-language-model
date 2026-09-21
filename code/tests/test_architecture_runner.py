"""Orchestration tests with mocked jobs; these are NOT experimental evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import sha
from scripts import run_architecture_screen as runner


class ArchitectureRunnerTests(unittest.TestCase):
    def exercise(self, allowed):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            baseline.write_bytes(b"mock-baseline-not-a-real-checkpoint")
            output = root / "screen"
            calls = []

            def fake_execute(arguments, log=None):
                values = list(map(str, arguments))
                calls.append(values)
                def arg(name):
                    return values[values.index(name) + 1]
                if values[0] == "scripts/benchmark_cpu.py":
                    Path(arg("--output")).write_text(json.dumps({
                        "within_five_x_time_limit": allowed,
                        "within_four_gib_peak_rss_limit": True,
                        "candidate_to_baseline_time_ratio": 4.0 if allowed else 6.0,
                        "candidate": {"max_peak_rss_bytes": 1000000},
                    }))
                elif values[0] == "train_experiment.py":
                    directory = Path(arg("--run-dir"))
                    directory.mkdir(exist_ok=True)
                    (directory / "checkpoint.pt").write_bytes(b"mock-endpoint")
                    (directory / "metrics.json").write_text(json.dumps({
                        "train_tokens": 58982400, "seed": 17, "train_seconds": 0.0}))
                elif values[0] == "scripts/average_checkpoints.py":
                    Path(arg("--output")).write_bytes(b"mock-average")
                elif values[0] == "evaluate.py":
                    self.assertEqual(arg("--split"), "validation")
                    Path(arg("--output")).write_text(json.dumps({
                        "checkpoint_sha256": sha(Path(arg("--checkpoint"))),
                        "targets": 376599, "bpb": 1.5,
                    }))
            with patch("sys.argv", ["screen", "--run-dir", str(output), "--baseline", str(baseline)]):
                with patch.object(runner, "execute", fake_execute):
                    runner.main()
            state = json.loads((output / "screen.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(state["candidates"]), 3)
            train_calls = [call for call in calls if call[0] == "train_experiment.py"]
            if not allowed:
                self.assertEqual(train_calls, [])
                self.assertTrue(all(row["status"] == "resource_rejected" for row in state["candidates"]))
            else:
                self.assertEqual(len(train_calls), 6)
                for first, resumed in zip(train_calls[::2], train_calls[1::2]):
                    self.assertEqual(first[-2:], ["--stop-after-step", "2"])
                    self.assertEqual(resumed[-1], "--resume")
                    self.assertEqual(first[:-2], resumed[:-1])
                self.assertTrue(all(row["final_resource_pass"] for row in state["candidates"]))
                self.assertEqual(sum(row["train_targets"] for row in state["candidates"]), 176947200)
                averages = [call for call in calls if call[0] == "scripts/average_checkpoints.py"]
                for call in averages:
                    sources = [Path(call[i+1]).name for i, x in enumerate(call) if x == "--checkpoint"]
                    self.assertEqual(sources, [f"step-{step:06d}.pt" for step in (6000,6300,6600,6900,7200)])

    def test_rejected_models_never_train(self):
        self.exercise(False)

    def test_fixed_budget_resume_and_validation_only_flow(self):
        self.exercise(True)

    def test_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            baseline.write_bytes(b"mock")
            output = root / "screen"
            with patch("sys.argv", ["screen", "--run-dir", str(output), "--baseline", str(baseline)]):
                with patch.object(runner, "execute", side_effect=RuntimeError("simulated failure")):
                    with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                        runner.main()
            self.assertEqual(json.loads((output / "screen.json").read_text())["status"], "failed")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("sys.argv", ["screen", "--run-dir", temporary, "--baseline", "missing"]):
                with self.assertRaises(SystemExit):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
