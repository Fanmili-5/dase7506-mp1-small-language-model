"""Small synthetic control tests; these weights are never used in experiments."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_experiment as trainer
import train as supplied_trainer


class TrainingResumeTests(unittest.TestCase):
    def arguments(self, directory, config, **overrides):
        values = dict(
            implementation="student", config=config, run_dir=directory,
            device="cpu", precision="fp32", threads=1, seed=17, steps=4,
            micro_batch_size=1, grad_accum=1, learning_rate=0.001,
            min_lr_ratio=0.1, warmup_steps=1, schedule="baseline",
            weight_decay=0.1, beta1=0.9, beta2=0.999, grad_clip=1.0,
            log_every=1, eval_every=2, eval_batch_size=1, save_every=2,
            keep_eval_checkpoints=False,
            resume=False, stop_after_step=0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def run_training(self, args):
        # Deterministic synthetic tokens for the test only, not external text.
        data = {
            "train": (torch.arange(1025) % 2048, 2048),
            "validation": (torch.arange(257) % 2048, 512),
        }
        with patch.object(trainer, "parse_args", return_value=args), \
             patch.object(trainer, "load_data", return_value=data), \
             contextlib.redirect_stdout(io.StringIO()):
            trainer.main()

    def test_resume_matches_uninterrupted_and_preserves_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tiny.json"
            config.write_text(json.dumps(dict(
                vocab=2048, context=256, width=16, heads=2, depth=1,
                dropout=0.1, position="rope", activation="swiglu",
            )), encoding="utf-8")
            full, resumed = root / "full", root / "resumed"
            self.run_training(self.arguments(full, config))
            self.run_training(self.arguments(resumed, config, stop_after_step=2))
            state = torch.load(resumed / "resume.pt", weights_only=False)
            self.assertEqual(state["step"], 2)
            self.assertFalse((resumed / "checkpoint.pt").exists())
            self.run_training(self.arguments(resumed, config, resume=True))
            expected = torch.load(full / "checkpoint.pt", weights_only=True)
            actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
            self.assertEqual(actual["train_tokens"], 4 * 256)
            for name, weight in expected["model"].items():
                torch.testing.assert_close(weight, actual["model"][name], atol=0, rtol=0)
            metrics = json.loads((resumed / "metrics.json").read_text())
            self.assertGreaterEqual(metrics["train_seconds"], state["accounted_train_seconds"])
            self.assertGreater(metrics["process_seconds"], state["accounted_process_seconds"])
            self.assertEqual(metrics["best_validation"]["bpb"],
                             min(row["bpb"] for row in metrics["validation_history"]))
            best = torch.load(resumed / "checkpoint-best.pt", weights_only=True)
            self.assertEqual(best["train_tokens"], metrics["best_validation"]["step"] * 256)
            with self.assertRaisesRegex(ValueError, "saved training plan"):
                self.run_training(self.arguments(resumed, config, resume=True, learning_rate=0.002))

    def test_baseline_learning_rate_matches_supplied_recipe(self):
        import math
        for step in (0, 50, 99, 100, 600, 1199):
            expected = 0.001 * min(1.0, (step + 1) / 100) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / 1200)))
            self.assertAlmostEqual(trainer.learning_rate(step, 1200, 0.001, 100, 0.1, "baseline"), expected)

    def test_periodic_checkpoints_can_be_preserved_for_averaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tiny.json"
            config.write_text(json.dumps(dict(
                vocab=2048, context=256, width=8, heads=1, depth=1,
            )), encoding="utf-8")
            run = root / "snapshots"
            self.run_training(self.arguments(
                run, config, steps=2, eval_every=1, save_every=2,
                keep_eval_checkpoints=True,
            ))
            first = torch.load(run / "checkpoints/step-000001.pt", weights_only=True)
            second = torch.load(run / "checkpoints/step-000002.pt", weights_only=True)
            self.assertEqual(first["train_tokens"], 256)
            self.assertEqual(second["train_tokens"], 512)

    def test_control_training_matches_supplied_trainer(self):
        # Test-only synthetic input; no resulting weights enter real experiments.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tiny-control.json"
            config.write_text(json.dumps(dict(
                vocab=2048, context=256, width=8, heads=1, depth=1,
            )), encoding="utf-8")
            supplied, extended = root / "supplied", root / "extended"
            args = self.arguments(extended, config, implementation="model",
                                  steps=100, warmup_steps=100, eval_every=0,
                                  save_every=100, log_every=100)
            self.run_training(args)
            official_args = SimpleNamespace(
                implementation="model", config=config, run_dir=supplied,
                device="cpu", precision="fp32", threads=1, seed=17,
                steps=100, batch_size=1, eval_every=0,
            )
            data = {
                "train": (torch.arange(1025) % 2048, 2048),
                "validation": (torch.arange(257) % 2048, 512),
            }
            with patch.object(supplied_trainer.argparse.ArgumentParser, "parse_args",
                              return_value=official_args), \
                 patch.object(supplied_trainer, "load_data", return_value=data), \
                 contextlib.redirect_stdout(io.StringIO()):
                supplied_trainer.main()
            expected = torch.load(supplied / "checkpoint.pt", weights_only=True)
            actual = torch.load(extended / "checkpoint.pt", weights_only=True)
            self.assertEqual(expected["train_tokens"], actual["train_tokens"])
            for name, weight in expected["model"].items():
                torch.testing.assert_close(weight, actual["model"][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
