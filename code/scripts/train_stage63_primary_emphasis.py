"""Late primary-emphasis continuation from the fixed Stage61 training average."""
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

START_SHA = "cd247efacddb28b6b55a6270dd202fee54a8c48ca8ad94d402c86df7fe31a2d8"
STEPS = 3600
PRIOR_STEPS = 16800
BATCH = 32
TARGETS = STEPS * BATCH * 256
AVERAGE_STEPS = (2400, 2700, 3000, 3300, 3600)
AUX_ANNEAL_STEPS = 600
AUX_START = .2
AUX_END = .05
CONFIG = Path("configs/stage54_hybrid_conv_rdrop.json")
SOURCE_FILES = (
    "student_hybrid_conv_rdrop.py", "student_hybrid_conv_structured.py",
    "student_rdrop_multi_token.py", "student_multi_token.py",
    "student_deep_supervision.py", "student_regularized.py",
    "student_structured.py", "student.py", "scripts/train_stage63_primary_emphasis.py",
    "train_experiment.py", "evaluate.py", "common.py", CONFIG.as_posix(),
    "data/manifest.json", "data/tokenizer.json",
)


def auxiliary_weight(step: int) -> float:
    progress = min(1.0, (step + 1) / AUX_ANNEAL_STEPS)
    return AUX_START + (AUX_END - AUX_START) * progress


def main():
    global START_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fresh-run", action="store_true",
                        help="Accept newly trained inputs; keep protocol checks and record actual hashes.")
    args = parser.parse_args()
    if args.fresh_run:
        START_SHA = sha(args.start)
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    if sha(args.start) != START_SHA:
        raise ValueError("Unexpected Stage61 averaged training checkpoint")
    start_payload = torch.load(args.start, map_location="cpu", weights_only=True)
    if (start_payload.get("protocol") != PROTOCOL
            or start_payload.get("implementation") != "student_hybrid_conv_rdrop"):
        raise ValueError("Expected the Stage61 R-Drop training average")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    if start_payload["config"] != config:
        raise ValueError("Stage61 configuration changed")
    if (config["deep_supervision_weight"] != AUX_START
            or config["future_prediction_weight"] != AUX_START):
        raise ValueError("Unexpected starting auxiliary weights")
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    model.load_state_dict(start_payload["model"], strict=True)
    torch.manual_seed(63017); torch.cuda.manual_seed_all(63017)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5,
                                  betas=(.9, .999), weight_decay=.1)
    data = load_data(); tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    for _ in range(PRIOR_STEPS):
        torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"]); span = 256 + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", start_checkpoint_sha256=START_SHA,
        start_stage="Stage61 fixed five-checkpoint training average",
        seed=17, dropout_rng_seed=63017, prior_sampling_steps=PRIOR_STEPS,
        continuation_steps=STEPS, additional_primary_targets=TARGETS,
        primary_stochastic_presentations=2 * TARGETS,
        deep_supervision_label_presentations=2 * TARGETS * len(
            config["deep_supervision_layers"]),
        future_label_presentations=2 * TARGETS * len(offsets),
        future_offsets=list(offsets), rdrop_alpha=config["rdrop_alpha"],
        stochastic_forwards_per_window=2,
        deep_supervision_weight_schedule=dict(start=AUX_START, end=AUX_END,
                                              linear_updates=AUX_ANNEAL_STEPS),
        future_prediction_weight_schedule=dict(start=AUX_START, end=AUX_END,
                                               linear_updates=AUX_ANNEAL_STEPS),
        optimizer="fresh AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=5e-5, minimum_learning_rate=5e-6, warmup_steps=50,
        comparison="late primary-emphasis continuation of Stage61 average",
        precision=precision, parameters=sum(p.numel() for p in model.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    started = time.perf_counter(); before = time.perf_counter()
    initial = score(model, *data["validation"], device, "fp32", 32)
    validation_seconds = time.perf_counter() - before
    initial.pop("window_nll_nats")
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)
    for step in range(STEPS):
        lr = learning_rate(step, STEPS, 5e-5, 50, .1, "warmup_cosine")
        weight = auxiliary_weight(step)
        model.deep_supervision_weight = weight
        model.future_prediction_weight = weight
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, primary_targets = extended[:, :256], extended[:, 1:257]
        future_targets = torch.stack([extended[:, offset:offset + 256]
                                      for offset in offsets])
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(ids, primary_targets, future_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step(); completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       primary_loss=float(parts["primary"]), deep_loss=float(parts["deep"]),
                       future_loss=float(parts["future"]),
                       symmetric_kl=float(parts["symmetric_kl"]), auxiliary_weight=weight,
                       learning_rate=lr, grad_norm=grad_norm,
                       additional_primary_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, auxiliary_weight=weight, **validation)
            validations.append(row); print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(checkpoint_payload(
                    model, "student_hybrid_conv_rdrop", config, 17,
                    (PRIOR_STEPS + completed) * BATCH * 256),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations,
                                  clamped_starts=int(clamped_starts)),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, 17,
        (PRIOR_STEPS + STEPS) * BATCH * 256), args.run_dir / "checkpoint.pt")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        additional_train_targets=TARGETS,
        cumulative_nominal_train_targets=(PRIOR_STEPS + STEPS) * BATCH * 256,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, clamped_starts=int(clamped_starts),
        history=history, validation_history=validations,
        final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"), **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
