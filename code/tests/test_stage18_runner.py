"""Synthetic orchestration checks, not measured model performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import sha
from scripts import run_stage18_regularization as runner


class Stage18RunnerTests(unittest.TestCase):
    def exercise(self, score):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline.pt"
            baseline.write_bytes(b"synthetic reference")
            previous = root / "previous.json"
            previous.write_text(json.dumps(dict(status="completed_teacher_rejected", baseline_sha256=runner.REFERENCE_SHA)))
            calls = []
            def fake_execute(arguments, log=None):
                command = list(map(str, arguments))
                calls.append(command)
                def arg(flag):
                    return command[command.index(flag) + 1]
                if command[0] == "train_experiment.py":
                    run = Path(arg("--run-dir"))
                    run.mkdir(exist_ok=True)
                    (run / "metrics.json").write_text(json.dumps(dict(train_tokens=runner.TARGETS, seed=17, train_seconds=0)))
                elif command[0] in ("scripts/average_checkpoints.py", "scripts/export_regularized.py"):
                    Path(arg("--output")).write_bytes(b"synthetic model")
                elif command[0] == "evaluate.py":
                    self.assertEqual(arg("--split"), "validation")
                    self.assertTrue(arg("--checkpoint").endswith("average-inference.pt"))
                    Path(arg("--output")).write_text(json.dumps(dict(bpb=score, targets=376599, checkpoint_sha256=sha(arg("--checkpoint")))))
                elif command[0] == "scripts/benchmark_cpu.py":
                    Path(arg("--output")).write_text(json.dumps(dict(candidate_to_baseline_time_ratio=4.8,
                        within_five_x_time_limit=True, within_four_gib_peak_rss_limit=True)))
            original_sha = runner.sha
            def fake_sha(path):
                return runner.REFERENCE_SHA if Path(path) == baseline else original_sha(path)
            argv = ["stage18", "--run-dir", str(root / "run"), "--baseline", str(baseline),
                    "--reference", str(baseline), "--stage17-screen", str(previous)]
            with patch("sys.argv", argv), patch.object(runner, "execute", fake_execute), patch.object(runner, "sha", fake_sha):
                runner.main()
            result = json.loads((root / "run/screen.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["actual_new_training_targets"], 117964800)
            train = [c for c in calls if c[0] == "train_experiment.py"]
            self.assertEqual(len(train), 4)
            for first, second in zip(train[::2], train[1::2]):
                self.assertEqual(first[:-2], second[:-1])
                self.assertEqual(first[-2:], ["--stop-after-step", "2"])
                self.assertEqual(second[-1], "--resume")
            benchmarks = [c for c in calls if c[0] == "scripts/benchmark_cpu.py"]
            self.assertEqual(len(benchmarks), 2 if score < 1.48 else 0)

    def test_fixed_equal_target_budget_export_before_scoring(self):
        self.exercise(1.47)

    def test_no_resource_retest_for_quality_rejection(self):
        self.exercise(1.50)


if __name__ == "__main__":
    unittest.main()
