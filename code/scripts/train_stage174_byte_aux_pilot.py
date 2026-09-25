"""Fixed 2,400-step same-target Stage54/Stage174 byte-supervision pilot."""
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
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)


CONFIG = Path("configs/stage174_byte_aux_rdrop.json")
IMPLEMENTATION = "student_stage174_byte_aux_rdrop"
SEED = 17
STEPS = 2400
BATCH = 32
CONTEXT = 256
PRIMARY_TARGETS = STEPS * BATCH * CONTEXT
CONTROL_2400_BPB = 1.519950369
REQUIRED_GAIN_BPB = 0.020
SOURCE_FILES = (
    "student_stage174_byte_aux_rdrop.py", "student_byte_composed.py",
    "student_hybrid_conv_rdrop.py", "student_hybrid_conv_structured.py",
    "student_rdrop_multi_token.py", "student_multi_token.py",
    "student_deep_supervision.py", "student_regularized.py",
    "student_structured.py", "student.py",
    "scripts/train_stage174_byte_aux_pilot.py", "train_experiment.py",
    "evaluate.py", "common.py", CONFIG.as_posix(),
    "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    args.run_dir.mkdir(parents=True)
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, implementation_sha = make_model(IMPLEMENTATION, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    data = load_data()
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"])
    span = CONTEXT + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        primary_targets=PRIMARY_TARGETS,
        primary_stochastic_presentations=2 * PRIMARY_TARGETS,
        future_label_presentations=2 * PRIMARY_TARGETS * len(offsets),
        byte_label_presentations=2 * PRIMARY_TARGETS * 2,
        method="Stage54 R-Drop plus training-only first/last ByteLevel CE",
        byte_aux_weight=config["token_byte_aux_weight"],
        control_2400_bpb=CONTROL_2400_BPB, required_gain_bpb=REQUIRED_GAIN_BPB,
        schedule_total_steps=7200, physical_batch=BATCH, effective_batch=BATCH,
        precision=precision,
        parameters_training=sum(p.numel() for p in model.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)
    for step in range(STEPS):
        lr = learning_rate(step, 7200, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, primary_targets = extended[:, :CONTEXT], extended[:, 1:CONTEXT + 1]
        future_targets = torch.stack([
            extended[:, offset:offset + CONTEXT] for offset in offsets])
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(ids, primary_targets, future_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(
                step=completed, loss=float(loss.detach()),
                primary_loss=float(parts["primary"]),
                deep_loss=float(parts["deep"]),
                future_loss=float(parts["future"]),
                symmetric_kl=float(parts["symmetric_kl"]),
                byte_aux_loss=float(parts["byte_aux"]),
                learning_rate=lr, grad_norm=grad_norm,
                primary_targets=completed * BATCH * CONTEXT,
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
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations,
                                  clamped_starts=int(clamped_starts)),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    checkpoint = args.run_dir / "checkpoint.pt"
    atomic_torch_save(checkpoint_payload(
        model, IMPLEMENTATION, config, SEED, PRIMARY_TARGETS), checkpoint)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    endpoint = validations[-1]
    gain = CONTROL_2400_BPB - endpoint["bpb"]
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        completed_steps=STEPS, train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, clamped_starts=int(clamped_starts),
        history=history, validation_history=validations,
        final_validation=endpoint, endpoint_gain_bpb=gain,
        continuation_gate_pass=gain >= REQUIRED_GAIN_BPB,
        checkpoint_sha256=sha(checkpoint), **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps({key: metrics[key] for key in (
        "status", "completed_steps", "final_validation", "endpoint_gain_bpb",
        "continuation_gate_pass", "checkpoint_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
