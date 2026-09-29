"""Train-only 900-step pilot distilling a fixed Stage105/155 teacher."""
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
from evaluate import score
from scripts.preflight_stage157_teacher import (STAGE105_SHA, STAGE155_SHA,
                                                 load_checkpoint)
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)

PREFLIGHT_SHA = "c8e2e5012334cb49e516dfa48e90f397c6a2439e12d227aad9320d62c85889ec"
INITIAL_BPB = 1.409877270416759
SEED = 157017
STEPS = 900
BATCH = 16
PEAK_LR = 2e-5
SOURCES = (
    "student_stage105_gated_singlepass.py", "student_stage104_gated_fast.py",
    "student_stage103_gated.py", "student_hybrid_conv_rdrop.py",
    "student_hybrid_conv_structured.py", "student_rdrop_multi_token.py",
    "student_multi_token.py", "student_deep_supervision.py",
    "student_regularized.py", "student_structured.py", "student.py",
    "student_ngram.py", "common.py", "evaluate.py", "train_experiment.py",
    "scripts/preflight_stage157_teacher.py",
    "scripts/train_stage157_complementarity_pilot.py",
    "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage105", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new training run directory")
    if sha(args.preflight) != PREFLIGHT_SHA:
        raise ValueError("Unexpected preflight record")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (not preflight.get("pilot_authorized")
            or preflight.get("stage105_sha256") != STAGE105_SHA
            or preflight.get("stage155_sha256") != STAGE155_SHA):
        raise ValueError("Stage157 pilot was not authorized by its preflight")
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    teacher_old, old_payload, old_module_sha = load_checkpoint(
        args.stage105, STAGE105_SHA, device)
    teacher_new, new_payload, new_module_sha = load_checkpoint(
        args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected frozen teacher implementation")
    student, student_module_sha = make_model(
        new_payload["implementation"], new_payload["config"], device)
    student.load_state_dict(new_payload["model"], strict=True)
    for teacher in (teacher_old, teacher_new):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=PEAK_LR,
                                  betas=(.9, .999), weight_decay=.1)
    data = load_data()
    tokens = data["train"][0]
    sampler = torch.Generator().manual_seed(SEED)
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    args.run_dir.mkdir(parents=True)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        physical_batch=BATCH, effective_batch=BATCH, context=256,
        primary_targets=STEPS * BATCH * 256,
        teacher_stage105_sha256=STAGE105_SHA,
        teacher_stage155_sha256=STAGE155_SHA,
        preflight_sha256=PREFLIGHT_SHA,
        teacher_stage105_module_sha256=old_module_sha,
        teacher_stage155_module_sha256=new_module_sha,
        student_implementation_sha256=student_module_sha,
        teacher_mixture_weights=[.5, .5], teacher_temperature=1.0,
        distillation_weight=.75, hard_nll_weight=.25,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR, warmup_steps=50,
        min_lr_ratio=.1, precision=precision,
        trainable_parameters=sum(p.numel() for p in student.parameters()),
        source_hashes=source_hashes,
        started_utc=datetime.now(timezone.utc).isoformat(),
        validation_labels_not_used_for_gradients=True,
        teacher_predictions_use_training_prefixes_only=True,
        no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    started = time.perf_counter()
    validation_seconds = 0.0
    student.eval()
    before = time.perf_counter()
    initial = score(student, *data["validation"], device, "fp32", 32)
    initial.pop("window_nll_nats")
    validation_seconds += time.perf_counter() - before
    if abs(initial["bpb"] - INITIAL_BPB) > 2e-5:
        raise ValueError(f"Stage157 initial validation mismatch: {initial['bpb']}")
    validations = [dict(step=0, **initial)]
    history = []
    print(json.dumps({"validation": validations[-1]}), flush=True)
    for step in range(STEPS):
        completed = step + 1
        rate = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = rate
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=sampler)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids = sequence[:, :256].to(device)
        targets = sequence[:, 1:].to(device)
        with torch.no_grad():
            old_logp = teacher_old.predict_log_probs(ids).float()
            new_logp = teacher_new.predict_log_probs(ids).float()
            teacher_probability = (torch.logaddexp(old_logp, new_logp)
                                   - math.log(2)).exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = student.predict_log_probs(ids).float()
            distill = -(teacher_probability * student_logp).sum(-1).mean()
            hard = F.nll_loss(student_logp.reshape(-1, 2048),
                              targets.reshape(-1))
            loss = .75 * distill + .25 * hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       distill_loss=float(distill.detach()),
                       hard_loss=float(hard.detach()), grad_norm=grad_norm,
                       learning_rate=rate,
                       primary_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            student.eval()
            before = time.perf_counter()
            result = score(student, *data["validation"], device, "fp32", 32)
            result.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            validations.append(dict(step=completed, **result))
            print(json.dumps({"validation": validations[-1]}), flush=True)
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    checkpoint = args.run_dir / "checkpoint.pt"
    atomic_torch_save(checkpoint_payload(student, new_payload["implementation"],
                                        new_payload["config"], SEED,
                                        STEPS * BATCH * 256), checkpoint)
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during pilot: {name}")
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        history=history, validation_history=validations,
        final_validation=validations[-1],
        pilot_gate_passed=INITIAL_BPB - validations[-1]["bpb"] >= .004,
        checkpoint_sha256=sha(checkpoint), **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
