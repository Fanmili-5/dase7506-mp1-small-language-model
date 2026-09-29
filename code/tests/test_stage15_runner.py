"""Mocked orchestration only; values here are not measured model results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import sha
from scripts import run_stage15_screen as runner


class Stage15RunnerTests(unittest.TestCase):
    def test_budget_averaging_and_asset_limits(self):
        self.assertEqual(sum(s[3] for s in runner.SPECS) * 8192, 353894400)
        self.assertEqual(runner.average_steps(7200), (6000, 6300, 6600, 6900, 7200))
        self.assertEqual(runner.average_steps(21600), (20400, 20700, 21000, 21300, 21600))
        with self.assertRaises(ValueError):
            runner.average_steps(7250)
        result = dict(within_five_x_time_limit=True, within_four_gib_peak_rss_limit=True)
        self.assertTrue(runner.resource_pass(result, 64 * 1024**2))
        self.assertFalse(runner.resource_pass(result, 64 * 1024**2 + 1))

    def exercise(self, allowed):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.pt"
            baseline.write_bytes(b"mock-checkpoint")
            directory = Path(temporary) / "screen"
            calls = []

            def fake_execute(arguments, log=None):
                command = list(map(str, arguments))
                calls.append(command)
                def arg(name):
                    return command[command.index(name) + 1]
                if command[0] == "scripts/benchmark_cpu.py":
                    Path(arg("--output")).write_text(json.dumps(dict(
                        within_five_x_time_limit=allowed, within_four_gib_peak_rss_limit=True,
                        candidate_to_baseline_time_ratio=4.0 if allowed else 6.0,
                        candidate=dict(max_peak_rss_bytes=1000000))))
                elif command[0] == "train_experiment.py":
                    run = Path(arg("--run-dir"))
                    run.mkdir(exist_ok=True)
                    (run / "checkpoint.pt").write_bytes(b"mock-endpoint")
                    (run / "metrics.json").write_text(json.dumps(dict(
                        train_tokens=int(arg("--steps")) * 8192, seed=17, train_seconds=0)))
                elif command[0] == "scripts/average_checkpoints.py":
                    Path(arg("--output")).write_bytes(b"mock-average")
                elif command[0] == "evaluate.py":
                    self.assertEqual(arg("--split"), "validation")
                    Path(arg("--output")).write_text(json.dumps(dict(
                        checkpoint_sha256=sha(arg("--checkpoint")), targets=376599, bpb=1.4)))
            real_sha = runner.sha
            def reference_sha(path):
                return runner.REFERENCE_SHA if Path(path) == baseline else real_sha(path)
            argv = ["stage15", "--baseline", str(baseline), "--reference", str(baseline),
                    "--run-dir", str(directory)]
            with patch("sys.argv", argv), patch.object(runner, "sha", reference_sha), patch.object(runner, "execute", fake_execute):
                runner.main()
            state = json.loads((directory / "screen.json").read_text())
            self.assertEqual(state["status"], "completed")
            training = [c for c in calls if c[0] == "train_experiment.py"]
            if allowed:
                self.assertEqual(len(training), 8)
                self.assertEqual(state["actual_new_training_targets"], 353894400)
                for first, second in zip(training[::2], training[1::2]):
                    self.assertEqual(first[-2:], ["--stop-after-step", "2"])
                    self.assertEqual(second[-1], "--resume")
                    self.assertEqual(first[:-2], second[:-1])
                self.assertTrue(all(r["final_resource_pass"] for r in state["candidates"]))
            else:
                self.assertFalse(training)
                self.assertEqual(state["actual_new_training_targets"], 0)

    def test_rejected_candidates_do_not_train(self):
        self.exercise(False)

    def test_fixed_budget_flow(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
