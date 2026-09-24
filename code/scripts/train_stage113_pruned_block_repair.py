"""Repair the seven-block Stage92 student on training prefixes only."""
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
from scripts.screen_stage112_block_ablation import (
    COUNTS_SHA, EXPECTED_BYTES, EXPECTED_TARGETS, score)
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import (
    atomic_json_dump, atomic_torch_save, checkpoint_payload, learning_rate,
    training_autocast)


IMPLEMENTATION = "student_hybrid_conv_pruned5"
SEED, STEPS, BATCH = 113017, 2400, 24
PEAK_LR = 3e-5
DISTILL_WEIGHT, HARD_WEIGHT = .75, .25
EVAL_STEPS = (0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400)
SAVE_STEPS = (900, 1200, 1500, 1800, 2100, 2400)
START_BPB = 1.4426014785352326
SOURCE_FILES = (
    "student_hybrid_conv_pruned5.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "student_mixture_aware.py",
    "scripts/analyze_stage98_distilled_gate.py",
    "scripts/screen_stage112_block_ablation.py",
    "scripts/train_stage113_pruned_block_repair.py", "train_experiment.py",
    "common.py", "evaluate.py", "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-start", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage113 run directory")
    if sha(args.student_start) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen Stage92 or order-six MKN checkpoint")
    neural_payload = torch.load(args.student_start, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected Stage113 ancestry")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    full, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    full.load_state_dict(neural_payload["model"], strict=True)
    full.eval()
    student, implementation_sha = make_model(IMPLEMENTATION, neural_payload["config"], device)
    warm_start, _ = make_model(neural_payload["implementation"],
                               neural_payload["config"], torch.device("cpu"))
    warm_start.load_state_dict(neural_payload["model"], strict=True)
    del warm_start.blocks[4]
    student.load_state_dict(warm_start.state_dict(), strict=True)
    del warm_start
    for parameter in full.parameters():
        parameter.requires_grad_(False)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True)
    counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    data = load_data()
    tokens = data["train"][0]
    validation, raw_bytes = data["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation bytes")
    batches = []
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            batches.append((ids, labels,
                            count_target_probability(counts, ids, labels, edge_keys)))
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1)
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", mechanism="remove_original_block5_then_repair",
        seed=SEED, steps=STEPS, batch_size=BATCH,
        new_training_targets=STEPS * targets_per_step,
        original_checkpoint_train_targets=int(neural_payload.get("train_tokens", 0)),
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR,
        teacher_cross_entropy_weight=DISTILL_WEIGHT,
        hard_next_token_nll_weight=HARD_WEIGHT,
        teacher_checkpoint_sha256=NEURAL_SHA,
        count_checkpoint_sha256=COUNTS_SHA,
        monitor="Stage94 fixed calibration plus Stage73 order-six MKN weight 0.0625",
        evaluation_steps=EVAL_STEPS, save_steps=SAVE_STEPS,
        training_prefixes_only=True, validation_labels_not_used_for_gradients=True,
        no_test_scoring=True, neural_training_device=str(device), precision=precision,
        implementation=IMPLEMENTATION, implementation_sha256=implementation_sha,
        source_hashes=sources, train_unigram_tokens=train_unigram_tokens,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()

    def evaluate(step):
        nonlocal validation_seconds
        student.eval()
        before = time.perf_counter()
        measured = score(student, batches, log_prior, raw_bytes)
        validation_seconds += time.perf_counter() - before
        if measured["targets"] != EXPECTED_TARGETS:
            raise ValueError("Validation coverage mismatch")
        row = dict(step=step, **measured)
        validations.append(row)
        print(json.dumps({"validation": row}), flush=True)
        return row

    initial = evaluate(0)
    if abs(initial["bpb"] - START_BPB) > 2e-5:
        raise ValueError("Pruned warm start does not reproduce Stage112")
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids = sequence[:, :256].to(device)
        labels = sequence[:, 1:].to(device)
        with torch.no_grad(), training_autocast(device, precision):
            teacher_probability = distilled_log_probs(full, ids, log_prior).exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = distilled_log_probs(student, ids, log_prior)
            distillation = -(teacher_probability * student_logp).sum(-1).mean()
            hard = F.nll_loss(student_logp.reshape(-1, student_logp.shape[-1]),
                              labels.reshape(-1))
            loss = DISTILL_WEIGHT * distillation + HARD_WEIGHT * hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite repair loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       teacher_cross_entropy=float(distillation.detach()),
                       hard_nll=float(hard.detach()), learning_rate=lr,
                       grad_norm=grad_norm,
                       new_training_targets=completed * targets_per_step,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed in EVAL_STEPS:
            evaluate(completed)
            if completed in SAVE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(student, IMPLEMENTATION,
                                       neural_payload["config"], SEED,
                                       int(neural_payload.get("train_tokens", 0))
                                       + completed * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
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
