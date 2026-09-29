"""Continue the Stage92 student using only the fixed heterogeneous teacher."""
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
from student_mixture_aware import build_target_edge_keys
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)


START_SHA = "3e8c7c04135faaa48947bc19ab68a3ea88fa26d036eb6b14ecfb9611516af724"
PRIMARY_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
ALTERNATE_SHA = "f4c499b63db9b39ce8062b4f07eff313ec23c491952e1b4c92f22d873b23bfdf"
PRIMARY_WEIGHT = .55
ALTERNATE_WEIGHT = .45
STEPS = 1500
PRIOR_UPDATES = 1800
BATCH = 24
PEAK_LR = 5e-6
SEED = 92017
AVERAGE_STEPS = (600, 900, 1200, 1500)
EXPECTED_INITIAL_BPB = 1.4017357354554512
SOURCE_FILES = (
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram.py",
    "student_mixture_aware.py", "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage93_pure_distillation.py", "train_experiment.py",
    "evaluate.py", "common.py", "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    expected = ((args.start, START_SHA), (args.primary, PRIMARY_SHA),
                (args.alternate, ALTERNATE_SHA), (args.counts, COUNTS_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Unexpected frozen checkpoint")
    start_payload = torch.load(args.start, map_location="cpu", weights_only=True)
    primary_payload = torch.load(args.primary, map_location="cpu", weights_only=True)
    alternate_payload = torch.load(args.alternate, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (start_payload.get("protocol") != PROTOCOL
            or start_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or primary_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or alternate_payload.get("implementation")
            != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected heterogeneous distillation payload")

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    student, implementation_sha = make_model(
        start_payload["implementation"], start_payload["config"], device)
    teacher_primary, _ = make_model(
        primary_payload["implementation"], primary_payload["config"], device)
    teacher_alternate, _ = make_model(
        alternate_payload["implementation"], alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    student.load_state_dict(start_payload["model"], strict=True)
    teacher_primary.load_state_dict(primary_payload["model"], strict=True)
    teacher_alternate.load_state_dict(alternate_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    teacher_primary.eval(); teacher_alternate.eval(); counts.eval()
    for model in (teacher_primary, teacher_alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior(); log_prior = log_prior.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1)
    data = load_data(); tokens = data["train"][0]
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(PRIOR_UPDATES):
        torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED,
        rng_draw_offset_updates=PRIOR_UPDATES, steps=STEPS, batch_size=BATCH,
        distillation_prefixes=STEPS * targets_per_step,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR,
        teacher_primary_weight=PRIMARY_WEIGHT,
        teacher_alternate_weight=ALTERNATE_WEIGHT,
        objective="pure_full_distribution_teacher_cross_entropy",
        hard_next_token_nll_weight=0.0, teacher_temperature=1.0,
        student_calibration=dict(temperature=1.10,
                                 train_unigram_prior_weight=.05,
                                 copy_gate_shift=.1875),
        validation_count_weight=.075,
        start_checkpoint_sha256=START_SHA,
        primary_checkpoint_sha256=PRIMARY_SHA,
        alternate_checkpoint_sha256=ALTERNATE_SHA,
        count_checkpoint_sha256=COUNTS_SHA,
        teacher_parameters_trainable=0,
        teacher_predictions_use_training_prefixes_only=True,
        next_token_labels_not_used_for_gradients=True,
        validation_labels_not_used_for_gradients=True,
        neural_training_device=str(device), precision=precision,
        student_parameters=sum(p.numel() for p in student.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.; started = time.perf_counter()
    student.eval(); before = time.perf_counter()
    initial = score_target_calibrated_mixture(
        student, counts, *data["validation"], device, "fp32", edge_keys,
        log_prior, 32)
    if abs(initial["bpb"] - EXPECTED_INITIAL_BPB) > 2e-6:
        raise ValueError("Initial student disagrees with Stage92")
    validation_seconds += time.perf_counter() - before
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)

    log_primary_weight = math.log(PRIMARY_WEIGHT)
    log_alternate_weight = math.log(ALTERNATE_WEIGHT)
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        ids = tokens[starts[:, None] + torch.arange(256)].to(device)
        with torch.no_grad(), training_autocast(device, precision):
            primary_logp = calibrated_neural_log_probs(
                teacher_primary, ids, log_prior)
            alternate_logp = teacher_alternate.predict_log_probs(ids)
            teacher_probability = torch.logaddexp(
                primary_logp + log_primary_weight,
                alternate_logp + log_alternate_weight).exp()
        optimizer.zero_grad(set_to_none=True); student.train()
        with training_autocast(device, precision):
            student_logp = calibrated_neural_log_probs(student, ids, log_prior)
            loss = -(teacher_probability * student_logp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(
                step=completed, distillation_cross_entropy=float(loss.detach()),
                learning_rate=lr, grad_norm=grad_norm,
                distillation_prefixes=completed * targets_per_step,
                train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            student.eval(); before = time.perf_counter()
            validation = score_target_calibrated_mixture(
                student, counts, *data["validation"], device, "fp32", edge_keys,
                log_prior, 32)
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row); print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(
                        student, start_payload["implementation"],
                        start_payload["config"], SEED,
                        (PRIOR_UPDATES + completed) * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(
                dict(completed_steps=completed, history=history,
                     validation_history=validations),
                args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(), history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
