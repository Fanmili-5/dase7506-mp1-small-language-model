"""Synthetic tests only; no synthetic weights are used in submitted predictors."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import train_distillation as kd
import train_experiment as ordinary
from common import ROOT, make_model, sha


class DistillationTests(unittest.TestCase):
    def test_kl_is_per_token_temperature_scaled_and_teacher_detached(self):
        s = torch.tensor([[[1., 2., -1.], [0., -1., 1.]]], requires_grad=True)
        t = torch.tensor([[[2., 0., -1.], [0., 1., 2.]]], requires_grad=True)
        y = torch.tensor([[1, 2]])
        loss, hard, soft = kd.distillation_loss(s, y, t, alpha=.4, temperature=2.)
        p = (t.detach() / 2).softmax(-1)
        q = (s / 2).softmax(-1)
        expected = 4 * (p * (p.log() - q.log())).sum() / 2
        torch.testing.assert_close(soft, expected)
        torch.testing.assert_close(loss, .6 * hard + .4 * expected)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        doubled = kd.distillation_loss(s.detach().repeat(2, 3, 1), y.repeat(2, 3),
                                      t.detach().repeat(2, 3, 1), alpha=.4, temperature=2.)
        torch.testing.assert_close(doubled[0], loss)

    def test_zero_alpha_is_exact_ce_and_identical_predictions_have_zero_kl(self):
        s = torch.randn(2, 3, 5, requires_grad=True)
        y = torch.zeros(2, 3, dtype=torch.long)
        loss, _, _ = kd.distillation_loss(s, y, alpha=0)
        self.assertTrue(torch.equal(loss, F.cross_entropy(s.flatten(0, 1), y.flatten())))
        loss, _, _ = kd.distillation_loss(s, y, s.detach(), alpha=1)
        self.assertEqual(float(loss), 0)
        loss.backward()
        torch.testing.assert_close(s.grad, torch.zeros_like(s.grad), atol=1e-7, rtol=0)

    def fixture(self, root):
        config = dict(vocab=2048, context=256, width=8, heads=1, depth=1, dropout=.1)
        path = root / "config.json"
        path.write_text(json.dumps(config))
        model, impl_sha = make_model("student", config, torch.device("cpu"))
        checkpoint = root / "teacher.pt"
        torch.save(ordinary.checkpoint_payload(model, "student", config, 17, 512), checkpoint)
        metrics = root / "teacher-metrics.json"
        metrics.write_text(json.dumps(dict(plan=dict(config=config), implementation_sha256=impl_sha,
            implementation="student", checkpoint_sha256=sha(checkpoint),
            data_manifest_sha256=sha(ROOT / "data/manifest.json"), train_tokens=512, train_seconds=.1)))
        return path, checkpoint, metrics

    def arguments(self, root, config, checkpoint, metrics, **overrides):
        with patch("sys.argv", ["train_distillation.py", "--config", str(config), "--run-dir", str(root)]):
            args = kd.parse_args()
        args.implementation = "student"
        args.steps, args.warmup_steps = 4, 1
        args.micro_batch_size, args.grad_accum, args.threads = 1, 1, 1
        args.eval_every, args.eval_batch_size, args.save_every, args.log_every = 2, 1, 2, 1
        args.teacher, args.teacher_sha256, args.teacher_metrics = checkpoint, sha(checkpoint), metrics
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def run_training(self, args, trainer=kd):
        data = {"train": (torch.arange(1025) % 2048, 2048),
                "validation": (torch.arange(257) % 2048, 512)}
        with patch.object(trainer, "parse_args", return_value=args), \
             patch.object(trainer, "load_data", return_value=data), \
             contextlib.redirect_stdout(io.StringIO()):
            trainer.main()

    def test_resume_exact_checkpoint_excludes_teacher_and_guards_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint, metrics = self.fixture(root)
            full = self.arguments(root / "full", config, checkpoint, metrics)
            resumed = self.arguments(root / "resumed", config, checkpoint, metrics, stop_after_step=2)
            self.run_training(full)
            self.run_training(resumed)
            resumed.resume, resumed.stop_after_step = True, 0
            self.run_training(resumed)
            expected = torch.load(full.run_dir / "checkpoint.pt", weights_only=True)
            actual = torch.load(resumed.run_dir / "checkpoint.pt", weights_only=True)
            self.assertFalse(any("teacher" in name for name in actual["model"]))
            self.assertEqual(actual["train_tokens"], 1024)
            self.assertEqual(actual["distillation_provenance"]["teacher"]["checkpoint_sha256"], sha(checkpoint))
            for key in actual["model"]:
                torch.testing.assert_close(actual["model"][key], expected["model"][key], atol=0, rtol=0)
            resumed.alpha = .7
            with self.assertRaisesRegex(ValueError, "saved training plan"):
                self.run_training(resumed)

    def test_ce_only_matches_original_trainer(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint, metrics = self.fixture(root)
            args = self.arguments(root / "kd", config, checkpoint, metrics, alpha=0, teacher=None,
                                  teacher_sha256=None, teacher_metrics=None)
            self.run_training(args)
            original = SimpleNamespace(**vars(args))
            original.run_dir = root / "ordinary"
            original.min_lr_ratio, original.schedule = .1, "baseline"
            original.weight_decay, original.beta1, original.beta2, original.grad_clip = .1, .9, .999, 1.
            original.keep_eval_checkpoints = True
            self.run_training(original, ordinary)
            a = torch.load(args.run_dir / "checkpoint.pt", weights_only=True)["model"]
            b = torch.load(original.run_dir / "checkpoint.pt", weights_only=True)["model"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_teacher_load_preserves_rng_frozen_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint, metrics = self.fixture(root)
            args = self.arguments(root / "run", config, checkpoint, metrics)
            before = torch.get_rng_state()
            teacher, _ = kd.load_teacher(args, torch.device("cpu"), sha(ROOT / "data/manifest.json"))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            args.teacher_sha256 = "bad"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                kd.load_teacher(args, torch.device("cpu"), sha(ROOT / "data/manifest.json"))

    def test_averaging_retains_teacher_provenance_and_rejects_mixed_teachers(self):
        from scripts.average_distilled_checkpoints import average_with_provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint, metrics = self.fixture(root)
            args = self.arguments(root / "full", config, checkpoint, metrics)
            self.run_training(args)
            paths = [args.run_dir / "checkpoints" / f"step-{step:06d}.pt" for step in (2, 4)]
            averaged = average_with_provenance(paths)
            self.assertEqual(averaged["train_tokens"], 1024)
            self.assertEqual(averaged["distillation_provenance"]["teacher"]["checkpoint_sha256"], sha(checkpoint))
            altered = torch.load(paths[1], weights_only=True)
            altered["distillation_provenance"]["teacher"]["checkpoint_sha256"] = "different"
            torch.save(altered, paths[1])
            with self.assertRaisesRegex(ValueError, "different distillation"):
                average_with_provenance(paths)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bf16_backward(self):
        model, _ = make_model("student_structured", dict(vocab=2048, width=32, heads=2,
            depth=1, context=256, output_kind="prefix_copy", copy_dim=8), torch.device("cuda"))
        ids = torch.arange(256, device="cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
            loss, _, _ = kd.distillation_loss(logits, ids, logits.detach() + .01, alpha=.5)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
