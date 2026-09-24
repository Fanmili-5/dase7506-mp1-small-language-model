"""Continue Stage92 by distilling the final neural/count probability mixture."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.screen_stage112_block_ablation import EXPECTED_BYTES, EXPECTED_TARGETS, score
from scripts.train_stage86_calibration_aware import calibrated_neural_log_probs, train_log_prior
from scripts.train_stage92_heterogeneous_distillation import PRIMARY_SHA, ALTERNATE_SHA
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)


SEED, STEPS, BATCH = 125017, 1500, 24
PEAK_LR = 3e-6
TEACHER_WEIGHTS = (.5375, .4250, .0375)
STUDENT_COUNT_WEIGHT = .0625
SAVE_STEPS = (900, 1200, 1500)
REFERENCE_BPB = 1.4017076627486786
SOURCE_FILES = (
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram.py",
    "student_mixture_aware.py", "scripts/analyze_stage98_distilled_gate.py",
    "scripts/screen_stage112_block_ablation.py",
    "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage92_heterogeneous_distillation.py",
    "scripts/train_stage125_mixture_aware_distillation.py",
    "train_experiment.py", "common.py", "evaluate.py",
    "data/manifest.json", "data/tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-start", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage125 run directory")
    expected = ((args.student_start, NEURAL_SHA), (args.primary, PRIMARY_SHA),
                (args.alternate, ALTERNATE_SHA), (args.counts, BASE_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Unexpected Stage125 frozen checkpoint ancestry")
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in expected]
    student_payload, primary_payload, alternate_payload, count_payload = payloads
    if (any(payload.get("protocol") != PROTOCOL for payload in payloads)
            or student_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or primary_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or alternate_payload.get("implementation") != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 5):
        raise ValueError("Unexpected Stage125 payload format")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    student, implementation_sha = make_model(
        student_payload["implementation"], student_payload["config"], device)
    primary, _ = make_model(primary_payload["implementation"], primary_payload["config"], device)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    for model, payload in zip((student, primary, alternate, counts), payloads):
        model.load_state_dict(payload["model"], strict=True)
    primary.eval(); alternate.eval(); counts.eval()
    for model in (primary, alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    data = load_data()
    train_tokens = data["train"][0]
    validation, raw_bytes = data["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation byte count")
    edge_keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        validation_batches = [
            (ids, labels, count_target_probability(counts, ids, labels, edge_keys))
            for ids, labels in windows(validation, 32)]
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1)
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED,
        mechanism="three_expert_teacher_to_final_neural_count_mixture",
        steps=STEPS, batch_size=BATCH,
        new_training_targets=STEPS * targets_per_step,
        prior_checkpoint_train_targets=int(student_payload.get("train_tokens", 0)),
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR,
        teacher_expert_weights=TEACHER_WEIGHTS,
        student_count_weight=STUDENT_COUNT_WEIGHT,
        teacher_cross_entropy_weight=.75, hard_final_mixture_nll_weight=.25,
        student_stage94_calibration=True,
        normalized_student_training_distribution=True,
        starting_checkpoint_sha256=NEURAL_SHA,
        primary_checkpoint_sha256=PRIMARY_SHA,
        alternate_checkpoint_sha256=ALTERNATE_SHA,
        count_checkpoint_sha256=BASE_SHA,
        teacher_predictions_use_train_prefixes_only=True,
        validation_labels_not_used_for_gradients=True,
        no_test_scoring=True, precision=precision, device=str(device),
        selected_average_steps=SAVE_STEPS,
        implementation_sha256=implementation_sha, source_hashes=sources,
        train_unigram_tokens=train_unigram_tokens,
        started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()

    def evaluate(step):
        nonlocal validation_seconds
        student.eval()
        before = time.perf_counter()
        measured = score(student, validation_batches, log_prior, raw_bytes)
        validation_seconds += time.perf_counter() - before
        if measured["targets"] != EXPECTED_TARGETS:
            raise ValueError("Validation target count mismatch")
        row = dict(step=step, **measured)
        validations.append(row)
        print(json.dumps({"validation": row}), flush=True)
        return row

    initial = evaluate(0)
    if abs(initial["bpb"] - REFERENCE_BPB) > 2e-5:
        raise ValueError("Stage92 start failed to reproduce Stage121 score")
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(train_tokens) - 257, (BATCH,), generator=rng)
        sequence = train_tokens[starts[:, None] + torch.arange(257)]
        ids_cpu, labels_cpu = sequence[:, :256], sequence[:, 1:]
        with torch.no_grad():
            count_logp = counts.predict_log_probs(ids_cpu).to(device)
        ids, labels = ids_cpu.to(device), labels_cpu.to(device)
        with torch.no_grad(), training_autocast(device, precision):
            primary_logp = calibrated_neural_log_probs(primary, ids, log_prior)
            alternate_logp = alternate.predict_log_probs(ids)
            teacher_neural = torch.logaddexp(
                primary_logp + math.log(TEACHER_WEIGHTS[0]),
                alternate_logp + math.log(TEACHER_WEIGHTS[1]))
            teacher_logp = torch.logaddexp(
                teacher_neural, count_logp + math.log(TEACHER_WEIGHTS[2]))
            teacher_probability = teacher_logp.exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = distilled_log_probs(student, ids, log_prior)
            final_logp = torch.logaddexp(
                student_logp + math.log1p(-STUDENT_COUNT_WEIGHT),
                count_logp + math.log(STUDENT_COUNT_WEIGHT))
            final_logp = final_logp - final_logp.logsumexp(-1, keepdim=True)
            distillation = -(teacher_probability * final_logp).sum(-1).mean()
            hard = F.nll_loss(final_logp.reshape(-1, final_logp.shape[-1]),
                              labels.reshape(-1))
            loss = .75 * distillation + .25 * hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Stage125 loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       teacher_cross_entropy=float(distillation.detach()),
                       hard_final_mixture_nll=float(hard.detach()),
                       learning_rate=lr, grad_norm=grad_norm,
                       new_training_targets=completed * targets_per_step,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            evaluate(completed)
            if completed in SAVE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(
                        student, student_payload["implementation"],
                        student_payload["config"], SEED,
                        int(student_payload.get("train_tokens", 0))
                        + completed * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during Stage125: " + name)
    metrics = dict(plan, status="completed_training_validation_only",
                   completed_utc=datetime.now(timezone.utc).isoformat(),
                   history=history, validation_history=validations,
                   final_validation=validations[-1],
                   best_validation=min(validations, key=lambda row: row["bpb"]),
                   train_seconds=time.perf_counter() - started - validation_seconds,
                   validation_seconds=validation_seconds, **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
