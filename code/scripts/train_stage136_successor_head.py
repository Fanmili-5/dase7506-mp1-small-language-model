"""Train only a causal successor-copy head on supplied train prefixes."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import (build_target_edge_keys,
                                   count_target_probability,
                                   mixture_target_log_probs)
from student_stage136_successor_neural import calibrated_successor_log_probs
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate)

SEED, STEPS, BATCH = 136017, 2400, 24
COUNT_WEIGHT, PEAK_LR = .0625, 5e-4
EVAL_STEPS = (0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400)
AVERAGE_STEPS = (1500, 1800, 2100, 2400)
IMPLEMENTATION = "student_stage136_successor_neural"
SOURCE_FILES = (
    "student_stage136_successor_neural.py", "student_stage135_successor_gate.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram.py",
    "student_mixture_aware.py", "scripts/train_stage136_successor_head.py",
    "scripts/train_stage86_calibration_aware.py", "train_experiment.py",
    "common.py", "evaluate.py", "data/manifest.json", "data/tokenizer.json",
)


def score(neural, batches, log_prior, byte_count):
    neural.eval()
    nll = 0.
    targets = 0
    gate_sum = 0.
    with torch.inference_mode():
        for ids_cpu, labels_cpu, count_cpu in batches:
            ids = ids_cpu.to(log_prior.device)
            labels = labels_cpu.to(log_prior.device)
            valid = labels != -100
            logp, gate = calibrated_successor_log_probs(
                neural, ids, log_prior, return_gate=True)
            count = count_cpu.to(log_prior.device)
            target_logp = mixture_target_log_probs(
                logp, count, labels, COUNT_WEIGHT)
            nll -= float(target_logp[valid].double().sum())
            gate_sum += float(gate.squeeze(-1)[valid].double().sum())
            targets += int(valid.sum())
    if log_prior.device.type == "cuda":
        torch.cuda.synchronize(log_prior.device)
    return dict(bpb=nll / math.log(2) / byte_count, nll_nats=nll,
                targets=targets, utf8_bytes=byte_count,
                mean_copy_gate=gate_sum / targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA:
        raise ValueError("Unexpected Stage92 neural or order-five count checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload["protocol"] != PROTOCOL
            or neural_payload["implementation"] != "student_hybrid_conv_output_bias"
            or count_payload["protocol"] != PROTOCOL
            or count_payload["implementation"] != "student_ngram"):
        raise ValueError("Wrong source checkpoint protocol or implementation")
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    neural, implementation_sha = make_model(IMPLEMENTATION, neural_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    neural.eval()  # freeze all dropout and all non-copy weights
    for parameter in neural.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for module in (neural.copy_query, neural.copy_key, neural.copy_gate):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=PEAK_LR, betas=(.9, .999),
                                  weight_decay=.01)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True)
    counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    data = load_data()
    tokens = data["train"][0]
    validation, raw_bytes = data["validation"]
    if raw_bytes != 1148007:
        raise ValueError("Unexpected validation bytes")
    batches = []
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            batches.append((ids, labels,
                            count_target_probability(counts, ids, labels, edge_keys)))
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    plan = dict(protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
                batch_size=BATCH, training_targets=STEPS * BATCH * 256,
                optimizer="AdamW", peak_lr=PEAK_LR, warmup_steps=100,
                weight_decay=.01, count_weight=COUNT_WEIGHT,
                trainable_parameters=trainable_count,
                trainable_modules=["copy_query", "copy_key", "copy_gate"],
                frozen_backbone_eval_mode=True,
                neural_start_sha256=NEURAL_SHA, count_sha256=BASE_SHA,
                implementation=IMPLEMENTATION, implementation_sha256=implementation_sha,
                source_hashes=sources, train_unigram_tokens=train_unigram_tokens,
                evaluation_steps=list(EVAL_STEPS), average_steps=list(AVERAGE_STEPS),
                training_prefixes_only=True, no_test_scoring=True,
                started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json_dump(plan, args.run_dir / "run.json")
    history = []
    validations = []
    rng = torch.Generator().manual_seed(SEED)
    started = time.perf_counter()
    for completed in range(STEPS + 1):
        if completed in EVAL_STEPS:
            row = dict(step=completed, **score(neural, batches, log_prior, raw_bytes))
            if row["targets"] != 376599:
                raise ValueError("Incomplete validation")
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed:
                payload = checkpoint_payload(
                    neural, IMPLEMENTATION, neural_payload["config"], SEED,
                    int(neural_payload.get("train_tokens", 0)) + completed * BATCH * 256)
                atomic_torch_save(payload, checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(dict(completed_steps=completed,
                                  validation_history=validations,
                                  train_history=history), args.run_dir / "progress.json")
        if completed == STEPS:
            break
        step = completed + 1
        lr = learning_rate(completed, STEPS, PEAK_LR, 100, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids_cpu, labels_cpu = sequence[:, :256], sequence[:, 1:]
        with torch.no_grad():
            count_target = count_target_probability(
                counts, ids_cpu, labels_cpu, edge_keys).to(device)
        ids = ids_cpu.to(device)
        labels = labels_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        logp = calibrated_successor_log_probs(neural, ids, log_prior)
        target_logp = mixture_target_log_probs(
            logp, count_target, labels, COUNT_WEIGHT)
        loss = -target_logp.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Stage136 loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        if step % 100 == 0:
            row = dict(step=step, loss=float(loss.detach()), lr=lr,
                       grad_norm=grad_norm,
                       training_targets=step * BATCH * 256,
                       elapsed_seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during Stage136: " + name)
    metrics = dict(plan, status="completed_training_validation_only",
                   completed_utc=datetime.now(timezone.utc).isoformat(),
                   validation_history=validations, train_history=history,
                   best_validation=min(validations, key=lambda row: row["bpb"]),
                   elapsed_seconds=time.perf_counter() - started,
                   **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(dict(status=metrics["status"],
                          best_validation=metrics["best_validation"],
                          elapsed_seconds=metrics["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
