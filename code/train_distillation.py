"""Training-only, same-data Transformer distillation; no teacher in inference assets.

Separate from the hash-pinned Stage15 trainer. Uses the same sampling, optimizer,
schedule, and hard-target objective as train_experiment when alpha is zero.
Distillation follows Hinton et al., https://arxiv.org/abs/1503.02531.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import ROOT, PROTOCOL, load_data, make_model, setup, sha, device_metrics
from evaluate import score
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)


def distillation_loss(student, targets, teacher=None, *, alpha=0.5, temperature=2.0):
    """KL is averaged over target positions, NOT just over batch items."""
    if not 0 <= alpha <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid distillation alpha or temperature")
    hard = F.cross_entropy(student.float().flatten(0, 1), targets.flatten())
    if alpha == 0:
        return hard, hard.detach(), hard.detach().new_zeros(())
    if teacher is None or teacher.shape != student.shape:
        raise ValueError("Teacher and student must supply matching full distributions")
    # Detach even when callers accidentally supply a differentiable teacher tensor.
    teacher_logp = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
    student_logp = F.log_softmax(student.float() / temperature, dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    soft = temperature ** 2 * kl
    return (1 - alpha) * hard + alpha * soft, hard.detach(), soft.detach()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--implementation", default="student_structured")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--teacher-sha256")
    p.add_argument("--teacher-metrics", type=Path)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=14400)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--save-every", type=int, default=300)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, default=0)
    return p.parse_args()


def load_teacher(args, device, data_sha):
    if args.alpha == 0:
        if args.teacher is not None:
            raise ValueError("CE-only control must not load a teacher")
        return None, None
    if not args.teacher or not args.teacher_sha256 or not args.teacher_metrics:
        raise ValueError("KD requires a pinned teacher and its training metrics")
    if sha(args.teacher) != args.teacher_sha256:
        raise ValueError("Teacher checkpoint hash mismatch")
    payload = torch.load(args.teacher, map_location="cpu", weights_only=True)
    metrics = json.loads(args.teacher_metrics.read_text(encoding="utf-8"))
    if payload["protocol"] != PROTOCOL or metrics["data_manifest_sha256"] != data_sha:
        raise ValueError("Teacher protocol/data provenance mismatch")
    if payload["config"] != metrics["plan"]["config"]:
        raise ValueError("Teacher config provenance mismatch")
    if payload["implementation"] != metrics["implementation"] or payload["train_tokens"] != metrics["train_tokens"]:
        raise ValueError("Teacher implementation/budget provenance mismatch")
    ancestry = payload.get("averaging_ancestry")
    if ancestry is None:
        if metrics["checkpoint_sha256"] != args.teacher_sha256:
            raise ValueError("Teacher metrics do not identify this checkpoint")
    else:
        targets = ancestry["source_train_targets"]
        per_step = 256 * metrics["plan"]["micro_batch_size"] * metrics["plan"]["grad_accum"]
        if (ancestry["method"] != "uniform_same_trajectory_parameter_average"
                or len(targets) != ancestry["source_count"] or not targets
                or max(targets) != payload["train_tokens"] or any(t % per_step for t in targets)):
            raise ValueError("Invalid teacher averaging ancestry")
        paths = [args.teacher.parent / "checkpoints" / f"step-{t // per_step:06d}.pt" for t in targets]
        if [sha(path) for path in paths] != ancestry["source_checkpoint_sha256"]:
            raise ValueError("Teacher source checkpoint hash mismatch")
    # Creating the teacher must not shift student dropout RNG relative to CE-only.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        teacher, impl_sha = make_model(payload["implementation"], payload["config"], device)
    if impl_sha != metrics["implementation_sha256"]:
        raise ValueError("Teacher implementation hash mismatch")
    teacher.load_state_dict(payload["model"])
    teacher.eval().requires_grad_(False)
    provenance = {
        "checkpoint_sha256": args.teacher_sha256,
        "metrics_sha256": sha(args.teacher_metrics),
        "implementation": payload["implementation"],
        "implementation_sha256": impl_sha,
        "teacher_training_targets": int(metrics["train_tokens"]),
        "teacher_training_seconds": float(metrics["train_seconds"]),
        "averaging_ancestry": ancestry,
    }
    return teacher, provenance


def main():
    args = parse_args()
    if any(getattr(args, key) <= 0 for key in (
            "steps", "micro_batch_size", "grad_accum", "eval_every", "eval_batch_size",
            "save_every", "log_every", "learning_rate", "threads")):
        raise ValueError("Training sizes, periods and learning rate must be positive")
    if not 0 <= args.warmup_steps <= args.steps or not 0 <= args.stop_after_step <= args.steps:
        raise ValueError("Warmup/stop step outside planned schedule")
    if not 0 <= args.alpha <= 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid distillation alpha or temperature")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise ValueError("Use an empty run directory or --resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = load_data()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, impl_sha = make_model(args.implementation, config, device)
    manifest_sha = sha(ROOT / "data/manifest.json")
    teacher, teacher_provenance = load_teacher(args, device, manifest_sha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    rng = torch.Generator().manual_seed(args.seed)
    plan = {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"resume", "stop_after_step", "run_dir", "teacher", "teacher_metrics", "config"}}
    plan.update(config=config, implementation_sha256=impl_sha,
                data_manifest_sha256=manifest_sha, teacher=teacher_provenance,
                torch_version=str(torch.__version__), precision=precision,
                sources={name: sha(ROOT / name) for name in (
                    "train_distillation.py", "train_experiment.py", "common.py", "evaluate.py",
                    "student.py", "student_structured.py")},
                optimizer="AdamW(beta1=.9,beta2=.999,wd=.1,clip=1)",
                schedule="baseline_cosine_min_ratio_.1")
    start, history, validations, accounted_seconds = 0, [], [], 0.0
    resume_path = args.run_dir / "resume.pt"
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["plan"] != plan:
            raise ValueError("Resume arguments do not match saved training plan")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, history, validations = state["step"], state["history"], state["validation_history"]
        accounted_seconds = state["accounted_seconds"]
        if args.stop_after_step and args.stop_after_step <= start:
            raise ValueError("Stop step must be after saved step")
    atomic_json_dump(plan, args.run_dir / "plan.json")
    tokens = data["train"][0].to(device)
    offsets = torch.arange(257, device=device)
    targets_per_step = 256 * args.micro_batch_size * args.grad_accum
    begun = time.perf_counter()

    def checkpoint(step):
        payload = checkpoint_payload(model, args.implementation, config, args.seed, step * targets_per_step)
        payload["distillation_provenance"] = {"alpha": args.alpha, "temperature": args.temperature,
                                               "teacher": teacher_provenance}
        return payload

    def save(step):
        atomic_torch_save(checkpoint(step), args.run_dir / "checkpoint-last.pt")
        atomic_torch_save(dict(plan=plan, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampling_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            history=history, validation_history=validations,
            accounted_seconds=accounted_seconds + time.perf_counter() - begun), resume_path)
        atomic_json_dump(dict(completed_steps=step, history=history, validation_history=validations),
                         args.run_dir / "progress.json")

    model.train()
    for step in range(start, args.steps):
        lr = learning_rate(step, args.steps, args.learning_rate, args.warmup_steps, 0.1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.grad_accum):
            starts = torch.randint(len(tokens) - 257, (args.micro_batch_size,), generator=rng).to(device)
            batch = tokens[starts[:, None] + offsets]
            with training_autocast(device, precision):
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(batch[:, :-1])
                logits = model(batch[:, :-1])
                loss, hard, soft = distillation_loss(logits, batch[:, 1:], teacher_logits,
                                                    alpha=args.alpha, temperature=args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss")
            (loss / args.grad_accum).backward()
            losses.append((float(loss.detach()), float(hard), float(soft)))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        completed = step + 1
        if completed % args.log_every == 0 or completed == args.steps:
            row = dict(step=completed, learning_rate=lr, grad_norm=norm,
                       train_targets=completed * targets_per_step,
                       **{key: sum(x[i] for x in losses) / len(losses)
                          for i, key in enumerate(("loss", "hard_ce", "soft_kl_T2"))})
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % args.eval_every == 0 or completed == args.steps:
            result = score(model, *data["validation"], device, "fp32", args.eval_batch_size)
            result.pop("window_nll_nats")
            if not math.isfinite(result["bpb"]):
                raise FloatingPointError("Non-finite validation")
            validations.append(dict(step=completed, **result))
            directory = args.run_dir / "checkpoints"
            directory.mkdir(exist_ok=True)
            snapshot = directory / f"step-{completed:06d}.pt"
            if snapshot.exists():
                raise FileExistsError(snapshot)
            atomic_torch_save(checkpoint(completed), snapshot)
            print(json.dumps({"validation": validations[-1]}), flush=True)
        stopping = completed == args.stop_after_step and completed < args.steps
        if completed % args.save_every == 0 or completed == args.steps or stopping:
            save(completed)
        if stopping:
            return
    atomic_torch_save(checkpoint(args.steps), args.run_dir / "checkpoint.pt")
    atomic_json_dump(dict(plan=plan, train_tokens=args.steps * targets_per_step,
        teacher_training_targets=0 if teacher_provenance is None else teacher_provenance["teacher_training_targets"],
        teacher_forward_targets=0 if teacher is None else args.steps * targets_per_step,
        process_seconds=accounted_seconds + time.perf_counter() - begun,
        cost_accounting="Includes saved student training and validation; teacher pretraining is separate. Crash-lost work excluded.",
        validation=validations[-1], history=history, validation_history=validations,
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"), **device_metrics(device)),
        args.run_dir / "metrics.json")


if __name__ == "__main__":
    main()
