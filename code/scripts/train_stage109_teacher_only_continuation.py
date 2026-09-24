"""Continue Stage92 with teacher-only train-prefix distillation, fixed budget."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from scripts.train_stage86_calibration_aware import (
    COUNTS_SHA, calibrated_neural_log_probs,
    score_target_calibrated_mixture, train_log_prior)
from scripts.train_stage92_heterogeneous_distillation import (
    ALTERNATE_SHA, PRIMARY_SHA, PRIMARY_WEIGHT, ALTERNATE_WEIGHT)
from student_mixture_aware import build_target_edge_keys
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)


STAGE92_SHA = "5e4ca423aeca652f5193f6b6b93eafe2a940ba6bab08689cb56b09c5c0b4c417"
STEPS, BATCH, SEED = 2400, 24, 109017
PEAK_LR = 5e-6
SAVE_STEPS = (1500, 1800, 2100, 2400)
START_BPB = 1.4017970556559118
SOURCE_FILES = (
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram.py",
    "student_mixture_aware.py", "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage92_heterogeneous_distillation.py",
    "scripts/train_stage109_teacher_only_continuation.py",
    "train_experiment.py", "evaluate.py", "common.py", "data/manifest.json",
    "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a fresh Stage109 run directory")
    expected_hashes = ((args.student, STAGE92_SHA), (args.primary, PRIMARY_SHA),
                       (args.alternate, ALTERNATE_SHA), (args.counts, COUNTS_SHA))
    if any(sha(path) != digest for path, digest in expected_hashes):
        raise ValueError("Unexpected frozen checkpoint ancestry")
    student_payload, primary_payload, alternate_payload, count_payload = (
        torch.load(path, map_location="cpu", weights_only=True)
        for path, _ in expected_hashes)
    if (any(payload.get("protocol") != PROTOCOL for payload in
            (student_payload, primary_payload, alternate_payload, count_payload))
            or student_payload["implementation"] != "student_hybrid_conv_output_bias"
            or student_payload["config"] != primary_payload["config"]
            or alternate_payload["implementation"] != "student_hybrid_conv_structured"
            or count_payload["implementation"] != "student_ngram"):
        raise ValueError("Unexpected model protocol or implementation")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    student, implementation_sha = make_model(
        student_payload["implementation"], student_payload["config"], device)
    primary, _ = make_model(
        primary_payload["implementation"], primary_payload["config"], device)
    alternate, _ = make_model(
        alternate_payload["implementation"], alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    student.load_state_dict(student_payload["model"], strict=True)
    primary.load_state_dict(primary_payload["model"], strict=True)
    alternate.load_state_dict(alternate_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    primary.eval(); alternate.eval(); counts.eval()
    for teacher in (primary, alternate, counts):
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1)
    data = load_data()
    tokens = data["train"][0]
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    prior_targets = int(student_payload.get("train_tokens", 0))
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED,
        mechanism="teacher_only_second_stage_heterogeneous_distillation",
        steps=STEPS, batch_size=BATCH, new_training_targets=STEPS * targets_per_step,
        prior_checkpoint_train_targets=prior_targets,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR, teacher_primary_weight=PRIMARY_WEIGHT,
        teacher_alternate_weight=ALTERNATE_WEIGHT,
        distillation_cross_entropy_weight=1.0, hard_next_token_nll_weight=0.0,
        student_checkpoint_sha256=STAGE92_SHA,
        primary_checkpoint_sha256=PRIMARY_SHA,
        alternate_checkpoint_sha256=ALTERNATE_SHA,
        count_checkpoint_sha256=COUNTS_SHA,
        fixed_monitor_calibration=dict(temperature=1.10,
                                       train_unigram_prior_weight=.05,
                                       copy_gate_shift=.1875, count_weight=.075),
        selected_average_steps=SAVE_STEPS,
        teacher_predictions_use_training_prefixes_only=True,
        validation_labels_not_used_for_gradients=True,
        no_test_scoring=True, neural_training_device=str(device), precision=precision,
        implementation_sha256=implementation_sha, source_hashes=sources,
        train_unigram_tokens=train_unigram_tokens,
        started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    student.eval()
    before = time.perf_counter()
    initial = score_target_calibrated_mixture(
        student, counts, *data["validation"], device, "fp32", edge_keys,
        log_prior, 32)
    if abs(initial["bpb"] - START_BPB) > 2e-6:
        raise ValueError("Stage92 starting validation did not reproduce")
    validation_seconds += time.perf_counter() - before
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    log_primary = math.log(PRIMARY_WEIGHT)
    log_alternate = math.log(ALTERNATE_WEIGHT)
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        ids = tokens[starts[:, None] + torch.arange(256)].to(device)
        with torch.no_grad(), training_autocast(device, precision):
            teacher_logp = torch.logaddexp(
                calibrated_neural_log_probs(primary, ids, log_prior) + log_primary,
                alternate.predict_log_probs(ids) + log_alternate)
            teacher_probability = teacher_logp.exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = calibrated_neural_log_probs(student, ids, log_prior)
            loss = -(teacher_probability * student_logp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite distillation loss at {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, distillation_cross_entropy=float(loss.detach()),
                       learning_rate=lr, grad_norm=grad_norm,
                       new_training_targets=completed * targets_per_step,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            student.eval()
            before = time.perf_counter()
            validation = score_target_calibrated_mixture(
                student, counts, *data["validation"], device, "fp32",
                edge_keys, log_prior, 32)
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in SAVE_STEPS:
                atomic_torch_save(checkpoint_payload(
                    student, student_payload["implementation"],
                    student_payload["config"], SEED,
                    prior_targets + completed * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        history=history, validation_history=validations,
        final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
