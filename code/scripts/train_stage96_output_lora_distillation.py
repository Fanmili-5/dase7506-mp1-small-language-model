"""Fit a frozen-backbone output LoRA to a fixed heterogeneous neural teacher.

Stage97 explicitly loads the primary teacher. Stage96 used the student object,
but its teacher path called the frozen tied head rather than the LoRA output
projection, so that earlier teacher was fixed as well. The Stage97 run source
is preserved in commit 043db25.
"""
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

from common import (PROTOCOL, device_metrics, load_data, make_model, setup,
                    sha, windows)
from scripts.train_stage86_calibration_aware import (
    COUNTS_SHA, calibrated_neural_log_probs, train_log_prior)
from student_mixture_aware import (build_target_edge_keys,
                                   count_target_probability,
                                   mixture_target_log_probs)
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload


PRIMARY_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
ALTERNATE_SHA = "f4c499b63db9b39ce8062b4f07eff313ec23c491952e1b4c92f22d873b23bfdf"
PRIMARY_WEIGHT, ALTERNATE_WEIGHT = .55, .45
DISTILL_WEIGHT, HARD_WEIGHT = .75, .25
EPOCHS, BATCH, LEARNING_RATE, SEED = 5, 32, .003, 96017
CONFIG = Path("configs/stage69_hybrid_conv_output_lora.json")
SOURCE_FILES = (
    "student_hybrid_conv_output_lora.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "student_mixture_aware.py",
    "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage96_output_lora_distillation.py", "train_experiment.py",
    "common.py", CONFIG.as_posix(), "data/manifest.json", "data/tokenizer.json",
)


def adapter_log_probs(model, ids, log_prior):
    hidden = model.features(ids)
    with torch.autocast(device_type=ids.device.type, enabled=False):
        hidden = hidden.float()
        vocabulary = F.log_softmax(
            (F.linear(hidden, model.output_weight()) + model.output_bias) / 1.10
            + .05 * log_prior, dim=-1)
        copy = model.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        gate = model.copy_gate(hidden) + .1875
        return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                               F.logsigmoid(gate) + log_copy)


def score_adapter(model, counts, tokens, byte_count, device, edge_keys,
                  log_prior, batch_size=32):
    model.eval(); counts.eval(); nll = 0.; target_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for ids, targets in windows(tokens, batch_size):
            count_probability = count_target_probability(
                counts, ids, targets, edge_keys).to(device)
            ids, targets = ids.to(device), targets.to(device)
            logp = adapter_log_probs(model, ids, log_prior)
            target_logp = mixture_target_log_probs(
                logp, count_probability, targets, .075)
            valid = targets != -100
            nll -= float(target_logp[valid].double().sum())
            target_count += int(valid.sum())
    torch.cuda.synchronize(device)
    return dict(bpb=nll / math.log(2) / byte_count,
                token_ppl=math.exp(nll / target_count), nll_nats=nll,
                targets=target_count, utf8_bytes=byte_count,
                seconds=time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    if (sha(args.primary) != PRIMARY_SHA or sha(args.alternate) != ALTERNATE_SHA
            or sha(args.counts) != COUNTS_SHA):
        raise ValueError("Unexpected frozen teacher checkpoint")
    primary_payload = torch.load(args.primary, map_location="cpu", weights_only=True)
    alternate_payload = torch.load(args.alternate, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    source_config = dict(config)
    source_config.pop("output_lora_rank"); source_config.pop("output_lora_alpha")
    if (primary_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or primary_payload["config"] != source_config
            or alternate_payload.get("implementation")
            != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected heterogeneous teacher payload")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    student, implementation_sha = make_model(
        "student_hybrid_conv_output_lora", config, device)
    incompatible = student.load_state_dict(primary_payload["model"], strict=False)
    if (incompatible.missing_keys != ["output_lora_a", "output_lora_b"]
            or incompatible.unexpected_keys):
        raise ValueError(f"Unexpected LoRA initialization: {incompatible}")
    primary, _ = make_model(primary_payload["implementation"],
                            primary_payload["config"], device)
    primary.load_state_dict(primary_payload["model"], strict=True)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    alternate.load_state_dict(alternate_payload["model"], strict=True)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.output_lora_a.requires_grad_(True)
    student.output_lora_b.requires_grad_(True)
    student.eval(); primary.eval(); alternate.eval(); counts.eval()
    for model in (primary, alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_unigram_tokens = train_log_prior(); log_prior = log_prior.to(device)
    optimizer = torch.optim.AdamW(
        [student.output_lora_a, student.output_lora_b], lr=LEARNING_RATE,
        betas=(.9, .999), weight_decay=.01)
    data = load_data(); train_tokens = data["train"][0]
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_epoch = len(train_tokens) - 1
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, epochs=EPOCHS,
        batch_size=BATCH, optimizer="AdamW", learning_rate=LEARNING_RATE,
        weight_decay=.01, output_lora_rank=config["output_lora_rank"],
        output_lora_alpha=config["output_lora_alpha"],
        teacher_primary_weight=PRIMARY_WEIGHT,
        teacher_alternate_weight=ALTERNATE_WEIGHT,
        distillation_cross_entropy_weight=DISTILL_WEIGHT,
        hard_next_token_nll_weight=HARD_WEIGHT,
        trained_parameters=(student.output_lora_a.numel()
                            + student.output_lora_b.numel()),
        frozen_backbone=True,
        fixed_primary_teacher=True,
        windowing="deterministic independent non-overlapping 256-token windows",
        initialization="Gaussian A and exact-zero B; exact Stage85 start",
        primary_checkpoint_sha256=PRIMARY_SHA,
        alternate_checkpoint_sha256=ALTERNATE_SHA,
        count_checkpoint_sha256=COUNTS_SHA,
        precision=precision, implementation_sha256=implementation_sha,
        source_hashes=sources, started_utc=datetime.now(timezone.utc).isoformat(),
        no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    started = time.perf_counter(); before = time.perf_counter()
    initial = score_adapter(student, counts, *data["validation"], device,
                            edge_keys, log_prior)
    validation_seconds = time.perf_counter() - before
    validations.append(dict(epoch=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    log_primary, log_alternate = math.log(PRIMARY_WEIGHT), math.log(ALTERNATE_WEIGHT)
    observed_targets = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.; epoch_targets = 0
        for ids_cpu, targets_cpu in windows(train_tokens, BATCH):
            valid = targets_cpu != -100
            ids, targets = ids_cpu.to(device), targets_cpu.to(device)
            with torch.no_grad():
                primary_logp = calibrated_neural_log_probs(primary, ids, log_prior)
                alternate_logp = alternate.predict_log_probs(ids)
                teacher_probability = torch.logaddexp(
                    primary_logp + log_primary,
                    alternate_logp + log_alternate).exp()
            optimizer.zero_grad(set_to_none=True)
            student_logp = adapter_log_probs(student, ids, log_prior)
            distillation = -(teacher_probability * student_logp).sum(-1)[valid].mean()
            hard_nll = F.nll_loss(student_logp.flatten(0, 1), targets.flatten(),
                                  ignore_index=-100)
            loss = DISTILL_WEIGHT * distillation + HARD_WEIGHT * hard_nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [student.output_lora_a, student.output_lora_b], 1.0)
            optimizer.step()
            count = int(valid.sum()); epoch_loss += float(loss.detach()) * count
            epoch_targets += count
        if epoch_targets != targets_per_epoch:
            raise ValueError("Training coverage changed")
        observed_targets += epoch_targets
        row = dict(epoch=epoch, mean_train_objective=epoch_loss / epoch_targets,
                   lora_a_l2=float(student.output_lora_a.detach().norm()),
                   lora_b_l2=float(student.output_lora_b.detach().norm()))
        history.append(row); print(json.dumps(row), flush=True)
        before = time.perf_counter()
        validation = score_adapter(student, counts, *data["validation"], device,
                                   edge_keys, log_prior)
        validation_seconds += time.perf_counter() - before
        validations.append(dict(epoch=epoch, **validation))
        print(json.dumps({"validation": validations[-1]}), flush=True)
        atomic_torch_save(
            checkpoint_payload(student, "student_hybrid_conv_output_lora",
                               config, SEED, observed_targets),
            checkpoints / f"epoch-{epoch:02d}.pt")
        atomic_json_dump(dict(completed_epochs=epoch, history=history,
                              validation_history=validations),
                         args.run_dir / "progress.json")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(plan, status="completed_training_validation_only",
                   completed_utc=datetime.now(timezone.utc).isoformat(),
                   history=history, validation_history=validations,
                   final_validation=validations[-1],
                   best_validation=min(validations, key=lambda row: row["bpb"]),
                   observed_train_targets=observed_targets,
                   train_seconds=time.perf_counter() - started - validation_seconds,
                   validation_seconds=validation_seconds, **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
