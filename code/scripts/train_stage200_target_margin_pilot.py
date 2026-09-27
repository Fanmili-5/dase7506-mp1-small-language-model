"""Fixed 2400-step, train/validation-only adaptive target-margin pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, make_model, setup, sha
from evaluate import score
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)

CONFIG = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
CONFIG_SHA = "87aeec0297b7f8b015276cdd51a5e3289de1ba0fef59f6c9845e69967d93fc39"
ALPHA = 0.005
SEED = 17
BATCH = 32
PILOT_STEPS = 2400
SCHEDULE_STEPS = 7200
CONTROL_BPB = 1.5199503686120217
ADMISSION_BPB = CONTROL_BPB - 0.03
SOURCE_FILES = (
    "student_hybrid_conv_rdrop.py", "student_rdrop_multi_token.py",
    "student_multi_token.py", "student_deep_supervision.py",
    "student_regularized.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "train_experiment.py",
    "evaluate.py", "common.py", "scripts/train_stage54_hybrid_conv_rdrop.py",
    "scripts/train_stage193_fresh_mixture_pilot.py",
    "scripts/train_stage200_target_margin_pilot.py",
    "tests/test_stage200_target_margin_pilot.py",
    "docs/STAGE200_ADVERSARIAL_TARGET_MARGIN_PLAN_20260928.md",
    "configs/stage54_hybrid_conv_rdrop.json", "data/manifest.json",
    "data/tokenizer.json", "data/wikitext_train.txt",
    "data/wikitext_validation.txt",
)


def target_margin_nll(logp: torch.Tensor, targets: torch.Tensor,
                      margin: torch.Tensor) -> torch.Tensor:
    """NLL when only target class logit is lowered by `margin`."""
    if logp.shape[:-1] != targets.shape or margin.shape != targets.shape:
        raise ValueError("Expected [batch,time,vocab] and [batch,time] tensors")
    if torch.any(margin < 0):
        raise ValueError("Margin must be nonnegative")
    target_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    probability = target_logp.exp()
    correction = torch.log1p(probability * torch.expm1(-margin))
    return (-target_logp + margin + correction).mean()


def margin_training_loss(model, ids: torch.Tensor, targets: torch.Tensor,
                         future_targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Change primary NLL only; keep auxiliary terms and unperturbed R-Drop KL."""
    hidden_norms: list[torch.Tensor] = []

    def capture_norm(_module, _inputs, output):
        hidden_norms.append(output.detach().float().norm(dim=-1))

    handle = model.norm.register_forward_hook(capture_norm)
    try:
        first_loss, first_logp, first_parts = model.single_training_pass(
            ids, targets, future_targets)
        second_loss, second_logp, second_parts = model.single_training_pass(
            ids, targets, future_targets)
    finally:
        handle.remove()
    if len(hidden_norms) != 2:
        raise RuntimeError("Expected one final hidden state per stochastic pass")
    with torch.autocast(device_type=ids.device.type, enabled=False):
        token_norm = model.head.weight.detach().float().norm(dim=-1)[targets]
        margins = [ALPHA * token_norm * hidden_norm for hidden_norm in hidden_norms]
        primary_losses = [
            target_margin_nll(logp.float(), targets, margin)
            for logp, margin in zip((first_logp, second_logp), margins)
        ]
        original_losses = [
            F.nll_loss(logp.flatten(0, 1), targets.flatten())
            for logp in (first_logp, second_logp)
        ]
        first_probability, second_probability = first_logp.exp(), second_logp.exp()
        symmetric_kl = 0.5 * (
            (first_probability * (first_logp - second_logp)).sum(dim=-1).mean()
            + (second_probability * (second_logp - first_logp)).sum(dim=-1).mean()
        )
        total = 0.5 * (first_loss - original_losses[0] + primary_losses[0]
                       + second_loss - original_losses[1] + primary_losses[1])
        total = total + model.rdrop_alpha * symmetric_kl
    parts = {
        "primary": 0.5 * (primary_losses[0].detach() + primary_losses[1].detach()),
        "deep": 0.5 * (first_parts["deep"] + second_parts["deep"]),
        "future": 0.5 * (first_parts["future"] + second_parts["future"]),
        "symmetric_kl": symmetric_kl.detach(),
        "mean_margin": 0.5 * (margins[0].mean() + margins[1].mean()),
    }
    return total, parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory; no result is overwritten")
    if sha(CONFIG) != CONFIG_SHA:
        raise ValueError("Stage54 configuration changed")
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    data = load_train_validation()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True)
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1)
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(SEED)
    plan = dict(
        protocol=PROTOCOL, stage="200", status="running", seed=SEED,
        alpha=ALPHA, batch=BATCH, context=256, pilot_steps=PILOT_STEPS,
        schedule_steps=SCHEDULE_STEPS, control_bpb=CONTROL_BPB,
        admission_bpb=ADMISSION_BPB, preflight_only=args.preflight,
        train_validation_only=True, config_sha256=CONFIG_SHA,
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    started = time.perf_counter()
    history = []
    clamped_starts = 0
    offsets = tuple(config["future_prediction_offsets"])
    span = 256 + max(offsets)
    steps = 1 if args.preflight else PILOT_STEPS
    for step in range(steps):
        lr = learning_rate(step, SCHEDULE_STEPS, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts += int((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, targets = extended[:, :256], extended[:, 1:257]
        future_targets = torch.stack([
            extended[:, offset:offset + 256] for offset in offsets
        ])
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = margin_training_loss(model, ids, targets, future_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage200 loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite Stage200 gradient at step {step + 1}")
        optimizer.step()
        if args.preflight or (step + 1) % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=step + 1, loss=float(loss.detach()),
                       primary_loss=float(parts["primary"]),
                       deep_loss=float(parts["deep"]),
                       future_loss=float(parts["future"]),
                       symmetric_kl=float(parts["symmetric_kl"]),
                       mean_margin=float(parts["mean_margin"]),
                       learning_rate=lr, grad_norm=grad_norm,
                       train_seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize(device)
    if any(sha(ROOT / name) != digest for name, digest in sources.items()):
        raise RuntimeError("Stage200 source changed during execution")
    if args.preflight:
        result = dict(plan, status="finite_one_update_preflight",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      elapsed_seconds=time.perf_counter() - started, history=history)
        atomic_json_dump(result, args.run_dir / "preflight.json")
        print(json.dumps(result | {"source_hashes": {}}, indent=2), flush=True)
        return
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, SEED,
        PILOT_STEPS * BATCH * 256), args.run_dir / "endpoint.pt")
    validation = score(model, *data["validation"], device, "fp32", BATCH)
    validation.pop("window_nll_nats")
    result = dict(
        plan, status="completed_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        primary_train_targets=PILOT_STEPS * BATCH * 256,
        validation=validation,
        admission_passed=validation["bpb"] <= ADMISSION_BPB,
        gain_vs_control_bpb=CONTROL_BPB - validation["bpb"],
        checkpoint_sha256=sha(args.run_dir / "endpoint.pt"),
        clamped_starts=clamped_starts, history=history,
        elapsed_seconds=time.perf_counter() - started,
        **device_metrics(device),
    )
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"history": [], "source_hashes": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
