"""Matched Stage47 R-Drop training for the Stage49-admitted hybrid backbone."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from train_experiment import (atomic_json_dump, atomic_torch_save, checkpoint_payload,
                              learning_rate, training_autocast)

STEPS = 7200
BATCH = 32
TARGETS = STEPS * BATCH * 256
AVERAGE_STEPS = (6000, 6300, 6600, 6900, 7200)
CONFIG = Path("configs/stage54_hybrid_conv_rdrop.json")
COMPARISON = "Stage47 matched R-Drop recipe; only backbone becomes Stage49 hybrid"
SOURCE_FILES = (
    "student_hybrid_conv_rdrop.py", "student_hybrid_conv_structured.py",
    "student_rdrop_multi_token.py", "student_multi_token.py",
    "student_deep_supervision.py", "student_regularized.py",
    "student_structured.py", "student.py", "scripts/train_stage54_hybrid_conv_rdrop.py",
    "train_experiment.py", "evaluate.py", "common.py", CONFIG.as_posix(),
    "data/manifest.json", "data/tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1)
    data = load_data(); tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"]); span = 256 + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=17, steps=STEPS,
        primary_targets=TARGETS, unique_primary_next_token_targets=TARGETS,
        primary_stochastic_presentations=2 * TARGETS,
        deep_supervision_label_presentations=2 * TARGETS * len(config["deep_supervision_layers"]),
        future_label_presentations=2 * TARGETS * len(offsets), future_offsets=list(offsets),
        rdrop_alpha=config["rdrop_alpha"], stochastic_forwards_per_window=2,
        deep_supervision_weight=config["deep_supervision_weight"],
        future_prediction_weight=config["future_prediction_weight"],
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        comparison=COMPARISON,
        precision=precision, parameters=sum(parameter.numel() for parameter in model.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.; started = time.perf_counter()
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)
    for step in range(STEPS):
        lr = learning_rate(step, STEPS, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups: group["lr"] = lr
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, primary_targets = extended[:, :256], extended[:, 1:257]
        future_targets = torch.stack([extended[:, offset:offset + 256] for offset in offsets])
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(ids, primary_targets, future_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        loss.backward(); grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step(); completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       primary_loss=float(parts["primary"]), deep_loss=float(parts["deep"]),
                       future_loss=float(parts["future"]), symmetric_kl=float(parts["symmetric_kl"]),
                       learning_rate=lr, grad_norm=grad_norm,
                       primary_targets=completed * BATCH * 256,
                       primary_stochastic_presentations=2 * completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation); validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(checkpoint_payload(
                    model, "student_hybrid_conv_rdrop", config, 17, completed * BATCH * 256),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations,
                                  clamped_starts=int(clamped_starts)), args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, 17, TARGETS), args.run_dir / "checkpoint.pt")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(), train_tokens=TARGETS,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, clamped_starts=int(clamped_starts),
        history=history, validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"), **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
