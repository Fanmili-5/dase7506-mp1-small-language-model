"""Mock orchestration tests; scores and timings here are NOT experimental data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import PROTOCOL, sha
from scripts import run_stage17_distillation as runner


class Stage17RunnerTests(unittest.TestCase):
    def screen(self):
        return dict(status="completed", protocol=PROTOCOL, split="validation",
                    reference_average_bpb=1.5,
                    candidates=[dict(average_bpb=1.48, final_resource_pass=True),
                                dict(average_bpb=1.3, final_resource_pass=False)])

    def test_reference_requires_completed_stage_and_qualified_predictor(self):
        screen = self.screen()
        self.assertEqual(runner.reference_from_stage15(screen), 1.48)
        screen["status"] = "training"
        with self.assertRaisesRegex(ValueError, "must complete"):
            runner.reference_from_stage15(screen)

    def exercise(self, teacher_pass):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            baseline.write_bytes(b"mock baseline")
            screen = self.screen() | {"baseline_sha256": sha(baseline)}
            previous = root / "previous.json"
            previous.write_text(json.dumps(screen))
            calls = []

            def execute(arguments, log=None):
                command = list(map(str, arguments))
                calls.append(command)
                def value(flag):
                    return command[command.index(flag) + 1]
                if command[0] in {"train_experiment.py", "train_distillation.py"}:
                    run = Path(value("--run-dir"))
                    run.mkdir(exist_ok=True)
                    (run / "metrics.json").write_text(json.dumps(dict(train_tokens=runner.TARGETS, process_seconds=0)))
                elif command[0] in {"scripts/average_checkpoints.py", "scripts/average_distilled_checkpoints.py"}:
                    self.assertEqual(command.count("--checkpoint"), 5)
                    self.assertIn("step-013200.pt", " ".join(command))
                    Path(value("--output")).write_bytes(b"mock average")
                elif command[0] == "evaluate.py":
                    self.assertEqual(value("--split"), "validation")
                    Path(value("--output")).write_text(json.dumps(dict(targets=376599,
                        checkpoint_sha256=sha(value("--checkpoint")), bpb=1.45 if teacher_pass else 1.5)))
                elif command[0] == "scripts/benchmark_cpu.py":
                    Path(value("--output")).write_text(json.dumps(dict(within_five_x_time_limit=True,
                        within_four_gib_peak_rss_limit=True, candidate_to_baseline_time_ratio=4.)))

            argv = ["stage17", "--run-dir", str(root / "run"), "--baseline", str(baseline),
                    "--stage15-screen", str(previous), "--execute"]
            with patch("sys.argv", argv), patch.object(runner, "execute", execute), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            result = json.loads((root / "run/screen.json").read_text())
            train = [c for c in calls if c[0] in {"train_experiment.py", "train_distillation.py"}]
            self.assertEqual(result["actual_new_gradient_targets"], runner.TARGETS * (3 if teacher_pass else 1))
            self.assertEqual(len(train), 6 if teacher_pass else 2)
            for smoke, resumed in zip(train[::2], train[1::2]):
                self.assertEqual(smoke[-2:], ["--stop-after-step", "2"])
                self.assertEqual(resumed[-1], "--resume")
                self.assertEqual(smoke[:-2], resumed[:-1])
            if teacher_pass:
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["kd_bpb_gain_over_matched_ce"], 0)
                self.assertNotIn("--teacher", train[2])
                self.assertIn("--teacher-sha256", train[4])
            else:
                self.assertEqual(result["status"], "completed_teacher_rejected")
                self.assertFalse(any(c[0] == "scripts/benchmark_cpu.py" for c in calls))

    def test_teacher_rejection_prevents_student_training(self):
        self.exercise(False)

    def test_fixed_matched_student_budget_and_no_test_scoring(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
