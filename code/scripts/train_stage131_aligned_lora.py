"""Repeat Stage129 with shuffled evaluator-aligned training windows only."""
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
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.screen_stage112_block_ablation import EXPECTED_BYTES, EXPECTED_TARGETS, score
from scripts.train_stage86_calibration_aware import calibrated_neural_log_probs, train_log_prior
from scripts.train_stage92_heterogeneous_distillation import PRIMARY_SHA, ALTERNATE_SHA
from scripts.train_stage129_backbone_lora import (
    RANK, SEED, BATCH, STEPS, PEAK_LR, WEIGHT_DECAY, SAVE_STEPS,
    REFERENCE_BPB, attach_lora, merged_model)
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)


SOURCE_FILES = (
    "scripts/train_stage131_aligned_lora.py",
    "scripts/train_stage129_backbone_lora.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "student_mixture_aware.py",
    "scripts/analyze_stage98_distilled_gate.py",
    "scripts/screen_stage112_block_ablation.py",
    "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage92_heterogeneous_distillation.py",
    "train_experiment.py", "common.py", "evaluate.py", "data/manifest.json",
    "data/tokenizer.json",
)


class AlignedWindowSampler:
    def __init__(self, tokens: torch.Tensor, batch: int, seed: int):
        self.batch = batch
        self.num_windows = (len(tokens) - 1) // 256
        if self.num_windows < batch:
            raise ValueError("Insufficient complete training windows")
        usable = self.num_windows * 256
        self.ids = tokens[:usable].reshape(self.num_windows, 256)
        self.targets = tokens[1:usable + 1].reshape(self.num_windows, 256)
        self.excluded_targets = len(tokens) - 1 - usable
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(self.num_windows, generator=self.rng)
        self.cursor = 0
        self.completed_epochs = 0

    def next(self):
        pieces = []
        remaining = self.batch
        while remaining:
            available = self.num_windows - self.cursor
            take = min(remaining, available)
            pieces.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.num_windows:
                self.completed_epochs += 1
                self.order = torch.randperm(self.num_windows, generator=self.rng)
                self.cursor = 0
        indices = torch.cat(pieces)
        return self.ids[indices], self.targets[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a fresh Stage131 directory")
    expected = ((args.student, NEURAL_SHA), (args.primary, PRIMARY_SHA),
                (args.alternate, ALTERNATE_SHA), (args.counts, BASE_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Stage131 frozen checkpoint ancestry changed")
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in expected]
    student_payload, primary_payload, alternate_payload, count_payload = payloads
    if (any(payload.get("protocol") != PROTOCOL for payload in payloads)
            or student_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or primary_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or alternate_payload.get("implementation") != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected Stage131 checkpoint implementation")
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    student, _ = make_model(student_payload["implementation"],
                            student_payload["config"], device)
    primary, _ = make_model(primary_payload["implementation"],
                            primary_payload["config"], device)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    for model, payload in zip((student, primary, alternate, counts), payloads):
        model.load_state_dict(payload["model"], strict=True)
    student.eval(); primary.eval(); alternate.eval(); counts.eval()
    for model in (primary, alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    prior, prior_tokens = train_log_prior()
    prior = prior.to(device)
    data = load_data()
    train_tokens = data["train"][0]
    validation, raw_bytes = data["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Validation bytes changed")
    sampler = AlignedWindowSampler(train_tokens, BATCH, SEED)
    keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        validation_batches = [
            (ids, labels, count_target_probability(counts, ids, labels, keys))
            for ids, labels in windows(validation, 32)]
    smoke_ids = (torch.arange(512, device=device).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        pre_attach = distilled_log_probs(student, smoke_ids, prior)
    names, trainable = attach_lora(student)
    with torch.inference_mode():
        attached = distilled_log_probs(student, smoke_ids, prior)
    initial_error = float((pre_attach.exp() - attached.exp()).abs().max())
    if initial_error > 3e-6:
        raise ValueError("Aligned LoRA zero-delta mismatch")
    merged_initial = merged_model(student)
    with torch.inference_mode():
        merged_logp = distilled_log_probs(merged_initial, smoke_ids, prior)
    initial_merge_error = float((attached.exp() - merged_logp.exp()).abs().max())
    if initial_merge_error > 3e-5:
        raise ValueError("Aligned LoRA initial merge mismatch")
    del merged_initial
    optimizer = torch.optim.AdamW(trainable, lr=PEAK_LR, betas=(.9, .999),
                                  weight_decay=WEIGHT_DECAY)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    initial = score(student, validation_batches, prior, raw_bytes)
    if (initial["targets"] != EXPECTED_TARGETS
            or abs(initial["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError("Stage131 step-0 control failed")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    plan = dict(
        protocol=PROTOCOL, status="training", mechanism="evaluator_aligned_lora_control",
        seed=SEED, rank=RANK, steps=STEPS, batch_size=BATCH,
        new_training_targets=STEPS * BATCH * 256,
        train_token_count=len(train_tokens),
        complete_aligned_train_windows=sampler.num_windows,
        excluded_final_partial_window_targets=sampler.excluded_targets,
        peak_learning_rate=PEAK_LR, weight_decay=WEIGHT_DECAY,
        trained_parameters=sum(p.numel() for p in trainable), adapted_linears=names,
        initial_probability_error=initial_error,
        initial_merge_probability_error=initial_merge_error,
        fixed_teacher_weights=[.55, .45], teacher_loss_weight=.75,
        hard_loss_weight=.25, selected_average_steps=SAVE_STEPS,
        source_hashes=sources, train_unigram_tokens=prior_tokens,
        validation_labels_not_used_for_gradients=True,
        teacher_uses_training_prefixes_only=True, no_test_scoring=True,
        precision=precision, device=str(device),
        started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json_dump(plan, args.run_dir / "run.json")
    print(json.dumps({"validation": dict(step=0, **initial)}), flush=True)
    history = []
    validations = [dict(step=0, **initial)]
    started = time.perf_counter()
    validation_seconds = 0.0
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids_cpu, labels_cpu = sampler.next()
        ids, labels = ids_cpu.to(device), labels_cpu.to(device)
        with torch.no_grad(), training_autocast(device, precision):
            primary_logp = calibrated_neural_log_probs(primary, ids, prior)
            alternate_logp = alternate.predict_log_probs(ids)
            teacher_logp = torch.logaddexp(primary_logp + math.log(.55),
                                           alternate_logp + math.log(.45))
            teacher_probability = teacher_logp.exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = distilled_log_probs(student, ids, prior)
            distillation = -(teacher_probability * student_logp).sum(-1).mean()
            hard = F.nll_loss(student_logp.flatten(0, 1), labels.flatten())
            loss = .75 * distillation + .25 * hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Stage131 loss at {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if completed == 1 and (not math.isfinite(grad_norm) or grad_norm <= 0):
            raise ValueError("Stage131 adapter gradients failed first-step preflight")
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       teacher_cross_entropy=float(distillation.detach()),
                       hard_nll=float(hard.detach()), learning_rate=lr,
                       grad_norm=grad_norm, completed_epochs=sampler.completed_epochs,
                       new_training_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            student.eval()
            before = time.perf_counter()
            measured = score(student, validation_batches, prior, raw_bytes)
            validation_seconds += time.perf_counter() - before
            if measured["targets"] != EXPECTED_TARGETS:
                raise ValueError("Stage131 validation target coverage mismatch")
            row = dict(step=completed, **measured)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in SAVE_STEPS:
                merged = merged_model(student)
                with torch.inference_mode():
                    adapted = distilled_log_probs(student, smoke_ids, prior)
                    merged_logp = distilled_log_probs(merged, smoke_ids, prior)
                merge_error = float((adapted.exp() - merged_logp.exp()).abs().max())
                if merge_error > 3e-5:
                    raise ValueError("Stage131 trained merge mismatch")
                payload = checkpoint_payload(
                    merged, student_payload["implementation"],
                    student_payload["config"], SEED,
                    int(student_payload.get("train_tokens", 0))
                    + completed * BATCH * 256)
                payload["merged_adapter_ancestry"] = dict(
                    source="stage131_evaluator_aligned_training", step=completed,
                    rank=RANK, start_sha256=NEURAL_SHA,
                    merge_max_probability_error=merge_error,
                    no_test_scoring=True)
                atomic_torch_save(payload, checkpoints / f"step-{completed:06d}.pt")
                del merged
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during Stage131: " + name)
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
