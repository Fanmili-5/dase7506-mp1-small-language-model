"""Reproducible MP1 trainer with accumulation, validation curves, and resume support.

This is an additive training utility. The supplied train.py and fixed evaluator
remain untouched. Checkpoints produced here retain the official evaluation
contract and can be scored directly with evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_autocast(device: torch.device, precision: str):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=precision in {"bf16", "fp16"})


def learning_rate(
    step: int,
    total_steps: int,
    peak: float,
    warmup: int,
    minimum_ratio: float,
    schedule: str,
) -> float:
    if schedule == "baseline":
        warmup_factor = min(1.0, (step + 1) / max(1, warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        return peak * warmup_factor * (minimum_ratio + (1.0 - minimum_ratio) * cosine)
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    decay_steps = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def checkpoint_payload(model, implementation: str, config: dict, seed: int, train_tokens: int) -> dict:
    return {
        "protocol": PROTOCOL,
        "implementation": implementation,
        "config": config,
        "model": model.state_dict(),
        "seed": seed,
        "train_tokens": train_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="student")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/student_control.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1200, help="Optimizer updates, including resumed updates.")
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--schedule", choices=["baseline", "warmup_cosine"], default="baseline")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int, default=0,
                        help="Save and stop early without changing the planned schedule (0 disables).")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "micro-batch-size": args.micro_batch_size,
        "grad-accum": args.grad_accum,
        "learning-rate": args.learning_rate,
        "log-every": args.log_every,
        "eval-batch-size": args.eval_batch_size,
        "save-every": args.save_every,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0, 1].")
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        raise ValueError("--warmup-steps must lie in [0, steps].")
    if args.stop_after_step < 0 or args.stop_after_step > args.steps:
        raise ValueError("--stop-after-step must lie in [0, steps].")


def main() -> None:
    total_started = time.perf_counter()
    args = parse_args()
    validate_args(args)

    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise ValueError("Run directory is not empty. Use a new directory or pass --resume.")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    requested_precision = args.precision
    setup_precision = "fp32" if requested_precision == "fp16" else requested_precision
    device, precision = setup(args.device, setup_precision, args.threads)
    if requested_precision == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 training is supported only on CUDA; use FP32 on CPU.")
        precision = "fp16"

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = load_data()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, implementation_sha = make_model(args.implementation, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)

    plan = {
        "resume_format": 2,
        "implementation": args.implementation,
        "implementation_sha256": implementation_sha,
        "trainer_sha256": sha(Path(__file__)),
        "data_manifest_sha256": sha(ROOT / "data/manifest.json"),
        "precision": precision,
        "device_type": device.type,
        "torch_version": str(torch.__version__),
        "threads": args.threads,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "config": config,
        "seed": args.seed,
        "steps": args.steps,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_steps": args.warmup_steps,
        "schedule": args.schedule,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "grad_clip": args.grad_clip,
    }

    start_step = 0
    history: list[dict] = []
    validation_history: list[dict] = []
    resumed_from = None
    best_validation = None
    previous_train_seconds = 0.0
    previous_validation_seconds = 0.0
    previous_process_seconds = 0.0
    resume_path = args.run_dir / "resume.pt"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume state: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["plan"] != plan:
            raise ValueError("Resume arguments do not match the saved training plan.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.set_state(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = int(state["step"])
        history = list(state["history"])
        validation_history = list(state["validation_history"])
        best_validation = state["best_validation"]
        previous_train_seconds = state["accounted_train_seconds"]
        previous_validation_seconds = state["accounted_validation_seconds"]
        previous_process_seconds = state["accounted_process_seconds"]
        resumed_from = str(resume_path)
        if args.stop_after_step and args.stop_after_step <= start_step:
            raise ValueError("--stop-after-step must be later than the saved step when resuming.")

    run_metadata = {
        "protocol": PROTOCOL,
        "implementation": args.implementation,
        "seed": args.seed,
        "command": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": str(torch.__version__),
        "requested_device": args.device,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "precision": precision,
        "threads": args.threads,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config_path": str(args.config),
        "config_sha256": sha(args.config),
        "implementation_sha256": implementation_sha,
        "trainer_sha256": plan["trainer_sha256"],
        "data_manifest_sha256": sha(ROOT / "data/manifest.json"),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_lr_ratio": args.min_lr_ratio,
            "warmup_steps": args.warmup_steps,
            "schedule": args.schedule,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "plan": plan,
        "targets_per_update": args.micro_batch_size * args.grad_accum * 256,
        "planned_train_targets": args.steps * args.micro_batch_size * args.grad_accum * 256,
        "resumed_from": resumed_from,
    }
    atomic_json_dump(run_metadata, args.run_dir / "run.json")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    validation_seconds = 0.0

    def train_elapsed() -> float:
        return previous_train_seconds + time.perf_counter() - started - validation_seconds

    def record_validation(completed_steps: int, validation: dict, kind: str) -> None:
        nonlocal best_validation
        if not math.isfinite(validation["bpb"]):
            raise FloatingPointError("Validation BPB is not finite.")
        row = {"step": completed_steps, "kind": kind, **validation}
        validation_history.append(row)
        if best_validation is None or row["bpb"] < best_validation["bpb"]:
            best_validation = row
            atomic_torch_save(
                checkpoint_payload(model, args.implementation, config, args.seed,
                                   completed_steps * args.micro_batch_size * args.grad_accum * 256),
                args.run_dir / "checkpoint-best.pt",
            )

    def save_state(completed_steps: int) -> None:
        train_targets = completed_steps * args.micro_batch_size * args.grad_accum * 256
        direct_checkpoint = checkpoint_payload(model, args.implementation, config, args.seed, train_targets)
        atomic_torch_save(direct_checkpoint, args.run_dir / "checkpoint-last.pt")
        resume_state = {
            "plan": plan,
            "step": completed_steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "sampling_rng": rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "history": history,
            "validation_history": validation_history,
            "best_validation": best_validation,
            "accounted_train_seconds": train_elapsed(),
            "accounted_validation_seconds": previous_validation_seconds + validation_seconds,
            "accounted_process_seconds": previous_process_seconds + time.perf_counter() - total_started,
        }
        atomic_torch_save(resume_state, resume_path)

    for step in range(start_step, args.steps):
        lr = learning_rate(
            step,
            args.steps,
            args.learning_rate,
            args.warmup_steps,
            args.min_lr_ratio,
            args.schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            starts = torch.randint(len(tokens) - 257, (args.micro_batch_size,), generator=rng).to(device)
            offsets = torch.arange(257, device=device)
            batch = tokens[starts[:, None] + offsets]
            with training_autocast(device, precision):
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())
                scaled_loss = loss / args.grad_accum
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at step {step + 1}.")
            scaler.scale(scaled_loss).backward()
            micro_losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        completed = step + 1
        if completed % args.log_every == 0 or completed == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "step": completed,
                "loss": sum(micro_losses) / len(micro_losses),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "train_targets": completed * args.micro_batch_size * args.grad_accum * 256,
                "seconds": train_elapsed(),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

        if args.eval_every > 0 and completed % args.eval_every == 0:
            before_validation = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", args.eval_batch_size)
            validation.pop("window_nll_nats")
            elapsed = time.perf_counter() - before_validation
            validation_seconds += elapsed
            record_validation(completed, validation, "periodic")
            print(json.dumps({"validation": validation_history[-1]}), flush=True)

        stopping_early = completed == args.stop_after_step and completed < args.steps
        if completed % args.save_every == 0 or completed == args.steps or stopping_early:
            save_state(completed)
            atomic_json_dump(
                {"completed_steps": completed, "history": history, "validation_history": validation_history},
                args.run_dir / "progress.json",
            )
        if stopping_early:
            print(json.dumps({"stopped_after_step": completed, "resume": str(resume_path)}), flush=True)
            return

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = train_elapsed()
    before_validation = time.perf_counter()
    final_validation = score(model, *data["validation"], device, "fp32", args.eval_batch_size)
    final_validation.pop("window_nll_nats")
    validation_seconds += time.perf_counter() - before_validation
    record_validation(args.steps, final_validation, "final")
    completed_targets = args.steps * args.micro_batch_size * args.grad_accum * 256
    final_checkpoint = args.run_dir / "checkpoint.pt"
    atomic_torch_save(
        checkpoint_payload(model, args.implementation, config, args.seed, completed_targets),
        final_checkpoint,
    )
    save_state(args.steps)
    result = {
        **run_metadata,
        "train_tokens": completed_targets,
        "train_seconds": train_seconds,
        "validation_seconds": previous_validation_seconds + validation_seconds,
        "validation": final_validation,
        "best_validation": best_validation,
        "best_checkpoint_sha256": sha(args.run_dir / "checkpoint-best.pt"),
        "history": history,
        "validation_history": validation_history,
        "checkpoint_sha256": sha(final_checkpoint),
        "checkpoint_bytes": final_checkpoint.stat().st_size,
        "process_seconds": previous_process_seconds + time.perf_counter() - total_started,
        "cost_accounting": "Cumulative through saved state; uncheckpointed work lost to a crash is not included.",
        **device_metrics(device),
    }
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"history": [], "validation_history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
