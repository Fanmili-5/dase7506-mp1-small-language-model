"""Fixed 2,400-step Stage54 comparison with quarter-batch SAM ascent."""
from __future__ import annotations

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
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.sam_training_step import sam_training_step
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
STEPS = 2400
FULL_SCHEDULE_STEPS = 7200
BATCH = 32
ASCENT_BATCH = 8
RHO = 0.05
TARGETS = STEPS * BATCH * 256
SOURCE_FILES = tuple(dict.fromkeys(base.SOURCE_FILES + (
    "scripts/sam_training_step.py",
    "scripts/run_stage168_sam_pilot.py",
)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    args.run_dir.mkdir(parents=True)
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / base.CONFIG).read_text(encoding="utf-8"))
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1)
    data = load_data()
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"])
    span = 256 + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=17, steps=STEPS,
        primary_targets=TARGETS, unique_primary_next_token_targets=TARGETS,
        descent_stochastic_presentations=2 * TARGETS,
        ascent_extra_stochastic_presentations=2 * STEPS * ASCENT_BATCH * 256,
        stochastic_forwards_per_descent_window=2,
        ascent_microbatch=ASCENT_BATCH, descent_batch=BATCH, sam_rho=RHO,
        full_schedule_steps=FULL_SCHEDULE_STEPS,
        optimizer="SAM(AdamW)", optimizer_betas=[.9, .999], weight_decay=.1,
        comparison="Stage54 same seed/windows/primary targets and LR; extra SAM ascent compute",
        precision=precision,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)
    for step in range(STEPS):
        lr = learning_rate(step, FULL_SCHEDULE_STEPS, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, primary_targets = extended[:, :256], extended[:, 1:257]
        future_targets = torch.stack([
            extended[:, offset:offset + 256] for offset in offsets
        ])

        def ascent_closure():
            with training_autocast(device, precision):
                return model.rdrop_training_loss(
                    ids[:ASCENT_BATCH], primary_targets[:ASCENT_BATCH],
                    future_targets[:, :ASCENT_BATCH])

        def descent_closure():
            with training_autocast(device, precision):
                return model.rdrop_training_loss(ids, primary_targets, future_targets)

        update = sam_training_step(
            model, optimizer, ascent_closure, descent_closure, rho=RHO)
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            parts = update["parts"]
            row = dict(
                step=completed, loss=float(update["loss"]),
                ascent_loss=float(update["ascent_loss"]),
                primary_loss=float(parts["primary"]),
                deep_loss=float(parts["deep"]),
                future_loss=float(parts["future"]),
                symmetric_kl=float(parts["symmetric_kl"]),
                learning_rate=lr,
                ascent_gradient_norm=update["ascent_gradient_norm"],
                descent_gradient_norm=update["descent_gradient_norm"],
                primary_targets=completed * BATCH * 256,
                descent_stochastic_presentations=2 * completed * BATCH * 256,
                ascent_extra_stochastic_presentations=2 * completed * ASCENT_BATCH * 256,
                train_seconds=time.perf_counter() - started - validation_seconds,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            atomic_json_dump(dict(
                completed_steps=completed, history=history,
                validation_history=validations,
                clamped_starts=int(clamped_starts)), args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, 17, TARGETS),
        args.run_dir / "checkpoint.pt")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        train_tokens=TARGETS,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        clamped_starts=int(clamped_starts), history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"),
        **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
