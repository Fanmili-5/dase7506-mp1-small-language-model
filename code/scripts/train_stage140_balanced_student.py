"""Distill the frozen Stage71/76 ensemble into the balanced Stage76 backbone."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from scripts.analyze_stage90_heterogeneous_ensemble import (
    ALTERNATE_SHA, PRIMARY_SHA,
)
from scripts.train_stage86_calibration_aware import (
    COUNTS_SHA, calibrated_neural_log_probs,
    score_target_calibrated_mixture, train_log_prior,
)
from student_mixture_aware import build_target_edge_keys
from train_experiment import (
    atomic_json_dump, atomic_torch_save, checkpoint_payload,
    learning_rate, training_autocast,
)


STEPS = 1800
BATCH = 24
SEED = 92017
PEAK_LR = 1e-5
AVERAGE_STEPS = (900, 1200, 1500, 1800)
PRIMARY_WEIGHT, ALTERNATE_WEIGHT = .55, .45
DISTILL_WEIGHT, HARD_WEIGHT = .75, .25
STUDENT_KIND = "student_hybrid_conv_output_bias"
SOURCE_FILES = (
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram.py",
    "student_mixture_aware.py", "scripts/train_stage140_balanced_student.py",
    "scripts/train_stage86_calibration_aware.py", "train_experiment.py",
    "evaluate.py", "common.py", "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Refusing to overwrite Stage140 run")
    if (sha(args.primary) != PRIMARY_SHA or sha(args.alternate) != ALTERNATE_SHA
            or sha(args.counts) != COUNTS_SHA):
        raise ValueError("Unexpected frozen teacher/count checkpoint")
    primary_payload = torch.load(args.primary, map_location="cpu", weights_only=True)
    alternate_payload = torch.load(args.alternate, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (primary_payload.get("protocol") != PROTOCOL
            or alternate_payload.get("protocol") != PROTOCOL
            or count_payload.get("protocol") != PROTOCOL
            or primary_payload.get("implementation") != STUDENT_KIND
            or alternate_payload.get("implementation")
            != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected source payload")
    student_config = dict(alternate_payload["config"], output_bias=True)
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    student, student_sha = make_model(STUDENT_KIND, student_config, device)
    teacher_primary, _ = make_model(primary_payload["implementation"],
                                    primary_payload["config"], device)
    teacher_alternate, _ = make_model(alternate_payload["implementation"],
                                      alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    missing = student.load_state_dict(alternate_payload["model"], strict=False)
    if missing.missing_keys != ["output_bias"] or missing.unexpected_keys:
        raise ValueError(f"Stage76 student initialization mismatch: {missing}")
    teacher_primary.load_state_dict(primary_payload["model"], strict=True)
    teacher_alternate.load_state_dict(alternate_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    student.eval(); teacher_primary.eval(); teacher_alternate.eval(); counts.eval()
    smoke_ids = (torch.arange(512, device=device).reshape(2, 256) * 31 + 7) % 2048
    with torch.inference_mode():
        zero_bias_error = float((student.predict_log_probs(smoke_ids).exp()
                                 - teacher_alternate.predict_log_probs(smoke_ids).exp())
                                .abs().max())
    if zero_bias_error > 3e-6:
        raise ValueError(f"Zero-bias Stage76 parity failed: {zero_bias_error}")
    for model in (teacher_primary, teacher_alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=PEAK_LR,
                                  betas=(.9, .999), weight_decay=.1)
    data = load_data()
    tokens = data["train"][0]
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        batch_size=BATCH, primary_targets=STEPS * targets_per_step,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR, teacher_primary_weight=PRIMARY_WEIGHT,
        teacher_alternate_weight=ALTERNATE_WEIGHT,
        distillation_cross_entropy_weight=DISTILL_WEIGHT,
        hard_next_token_nll_weight=HARD_WEIGHT,
        student_implementation=STUDENT_KIND, student_config=student_config,
        primary_sha256=PRIMARY_SHA, alternate_sha256=ALTERNATE_SHA,
        counts_sha256=COUNTS_SHA, zero_bias_probability_error=zero_bias_error,
        train_unigram_tokens=train_unigram_tokens,
        validation_calibration="Stage86 fixed", validation_count_weight=.075,
        teacher_parameters_trainable=0, validation_labels_not_used_for_gradients=True,
        implementation_sha256=student_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    before = time.perf_counter()
    initial = score_target_calibrated_mixture(
        student, counts, *data["validation"], device, "fp32", edge_keys,
        log_prior, 32)
    validation_seconds += time.perf_counter() - before
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    log_primary, log_alternate = math.log(PRIMARY_WEIGHT), math.log(ALTERNATE_WEIGHT)
    for step in range(STEPS):
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids = sequence[:, :256].to(device)
        targets = sequence[:, 1:].to(device)
        with torch.no_grad(), training_autocast(device, precision):
            primary_logp = calibrated_neural_log_probs(
                teacher_primary, ids, log_prior)
            alternate_logp = teacher_alternate.predict_log_probs(ids)
            teacher_probability = torch.logaddexp(
                primary_logp + log_primary,
                alternate_logp + log_alternate).exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = calibrated_neural_log_probs(student, ids, log_prior)
            distillation = -(teacher_probability * student_logp).sum(-1).mean()
            hard_nll = F.nll_loss(student_logp.reshape(-1, student_logp.shape[-1]),
                                  targets.reshape(-1))
            loss = DISTILL_WEIGHT * distillation + HARD_WEIGHT * hard_nll
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Stage140 loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       distillation_cross_entropy=float(distillation.detach()),
                       hard_next_token_nll=float(hard_nll.detach()),
                       learning_rate=lr, grad_norm=grad_norm,
                       primary_targets=completed * targets_per_step,
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
            if completed in AVERAGE_STEPS:
                atomic_torch_save(checkpoint_payload(
                    student, STUDENT_KIND, student_config, SEED,
                    completed * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during Stage140 training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        history=history, validation_history=validations,
        best_validation=min(validations, key=lambda row: row["bpb"]),
        final_validation=validations[-1],
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("history", "validation_history")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
