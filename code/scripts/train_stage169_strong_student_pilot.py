"""Matched teacher-versus-hard-label Stage169 train-only pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from evaluate import score
from scripts.preflight_stage157_teacher import STAGE105_SHA, STAGE155_SHA, load_checkpoint
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)

PREFLIGHT_CANONICAL_SHA = "81b1cd227c630e55c1cacca97741b70e302c6ded9ea331b3b584b0de7db7787e"
INITIAL_BPB = 1.399686162042141
SEED = 169017
STEPS = 900
BATCH = 8
PEAK_LR = 2e-5
SOURCES = (
    "student_stage103_gated.py", "student_stage105_gated_singlepass.py",
    "student_stage104_gated_fast.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_ngram_collapsed.py",
    "student_ngram.py", "student_structured.py", "student.py",
    "scripts/preflight_stage157_teacher.py",
    "scripts/preflight_stage169_strong_student.py",
    "scripts/train_stage169_strong_student_pilot.py", "evaluate.py",
    "common.py", "train_experiment.py", "data/manifest.json",
    "data/tokenizer.json",
)


def validate(model, data, device) -> dict:
    model.eval()
    result = score(model, *data["validation"], device, "fp32", BATCH)
    result.pop("window_nll_nats")
    return result


def run_arm(name: str, old_payload: dict, old, new, data: dict,
            device: torch.device, precision: str, output: Path,
            source_hashes: dict) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    student, student_module_sha = make_model("student_stage103_gated",
                                            old_payload["config"], device)
    student.load_state_dict(old_payload["model"], strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=PEAK_LR,
                                  betas=(.9, .999), weight_decay=.1)
    sampler = torch.Generator().manual_seed(SEED)
    tokens = data["train"][0]
    arm_dir = output / name
    arm_dir.mkdir()
    initial = validate(student, data, device)
    if abs(initial["bpb"] - INITIAL_BPB) > 2e-5:
        raise ValueError(f"Unexpected strong-student initial score: {initial['bpb']}")
    validations = [dict(step=0, **initial)]
    history = []
    validation_seconds = 0.0
    started = time.perf_counter()
    print(json.dumps({"arm": name, "validation": validations[-1]}), flush=True)
    for step in range(STEPS):
        completed = step + 1
        rate = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = rate
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=sampler)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids = sequence[:, :256].to(device)
        targets = sequence[:, 1:].to(device)
        if name == "teacher":
            with torch.no_grad():
                old_logp = old.predict_log_probs(ids).float()
                new_logp = new.predict_log_probs(ids).float()
                teacher_probability = (torch.logaddexp(old_logp, new_logp)
                                       - math.log(2)).exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = student.predict_log_probs(ids).float()
            hard = F.nll_loss(student_logp.reshape(-1, 2048), targets.reshape(-1))
            if name == "teacher":
                distill = -(teacher_probability * student_logp).sum(-1).mean()
                loss = .75 * distill + .25 * hard
            else:
                distill = None
                loss = hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss at {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       hard_loss=float(hard.detach()),
                       distill_loss=None if distill is None else float(distill.detach()),
                       grad_norm=grad_norm, learning_rate=rate,
                       primary_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps({"arm": name, "train": row}), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            result = validate(student, data, device)
            validation_seconds += time.perf_counter() - before
            validations.append(dict(step=completed, **result))
            print(json.dumps({"arm": name, "validation": validations[-1]}), flush=True)
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations), arm_dir / "progress.json")
    checkpoint = arm_dir / "checkpoint.pt"
    atomic_torch_save(checkpoint_payload(student, "student_stage103_gated",
                                        old_payload["config"], SEED,
                                        STEPS * BATCH * 256), checkpoint)
    for relative, expected in source_hashes.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Source changed during pilot: {relative}")
    metrics = dict(
        protocol=PROTOCOL, arm=name, seed=SEED, steps=STEPS,
        primary_targets=STEPS * BATCH * 256,
        initial_validation=validations[0], final_validation=validations[-1],
        validation_history=validations, history=history,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        checkpoint_sha256=sha(checkpoint),
        student_implementation_sha256=student_module_sha,
        source_hashes=source_hashes, precision=precision,
        no_test_scoring=True,
    )
    atomic_json_dump(metrics, arm_dir / "metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage105", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new pilot run directory")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(preflight, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != PREFLIGHT_CANONICAL_SHA:
        raise ValueError("Stage169 preflight evidence changed")
    if (not preflight.get("pilot_authorized")
            or preflight.get("stage105_sha256") != STAGE105_SHA
            or preflight.get("stage155_sha256") != STAGE155_SHA):
        raise ValueError("Stage169 feasibility gate did not pass")
    device, precision = setup("cuda", "bf16", 4)
    old, old_payload, old_module_sha = load_checkpoint(args.stage105, STAGE105_SHA, device)
    new, new_payload, new_module_sha = load_checkpoint(args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected teacher implementation")
    for teacher in (old, new):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    data = load_data()
    sources = {relative: sha(ROOT / relative) for relative in SOURCES}
    args.run_dir.mkdir(parents=True)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        arms=["teacher", "hard_only"], physical_batch=BATCH,
        effective_batch=BATCH, context=256,
        primary_targets_per_arm=STEPS * BATCH * 256,
        teacher_stage105_sha256=STAGE105_SHA,
        teacher_stage155_sha256=STAGE155_SHA,
        teacher_stage105_module_sha256=old_module_sha,
        teacher_stage155_module_sha256=new_module_sha,
        preflight_canonical_sha256=PREFLIGHT_CANONICAL_SHA,
        teacher_mixture_weights=[.5, .5], distillation_weight=.75,
        hard_nll_weight=.25, peak_learning_rate=PEAK_LR,
        warmup_steps=50, min_lr_ratio=.1,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        precision=precision, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(),
        validation_labels_not_used_for_gradients=True,
        teacher_predictions_use_training_prefixes_only=True,
        no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    teacher = run_arm("teacher", old_payload, old, new, data, device,
                      precision, args.run_dir, sources)
    control = run_arm("hard_only", old_payload, old, new, data, device,
                      precision, args.run_dir, sources)
    if abs(teacher["initial_validation"]["bpb"] - control["initial_validation"]["bpb"]) > 1e-10:
        raise ValueError("Matched controls did not start identically")
    teacher_bpb = teacher["final_validation"]["bpb"]
    control_bpb = control["final_validation"]["bpb"]
    result = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        teacher_final_bpb=teacher_bpb, control_final_bpb=control_bpb,
        teacher_gain_from_stage143=INITIAL_BPB - teacher_bpb,
        teacher_gain_over_hard_only=control_bpb - teacher_bpb,
        continuation_gate_passed=(teacher_bpb <= INITIAL_BPB - .005
                                  and teacher_bpb <= control_bpb - .003),
        teacher_checkpoint_sha256=teacher["checkpoint_sha256"],
        control_checkpoint_sha256=control["checkpoint_sha256"],
        **device_metrics(device),
    )
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"source_hashes": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
