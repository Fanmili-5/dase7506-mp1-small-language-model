"""Matched Stage195 train/validation pilot; never reads the test split."""
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

from common import PROTOCOL, device_metrics, setup, sha
from evaluate import score
from scripts.preflight_stage157_teacher import STAGE105_SHA, STAGE155_SHA, load_checkpoint
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from student_stage195_feature_residual import FeatureResidualGatedLM
from train_experiment import atomic_json_dump, learning_rate, training_autocast

SEED = 195017
STEPS = 1200
BATCH = 8
PEAK_LR = 3e-4
BASE_VALIDATION_BPB = 1.399686162042141
PREFLIGHT_SOURCE_SHA = "7a3d1f004065ca7d361bfbe8f859e0c9f20b88818a8ee749bf6b547573f0cf1e"
SOURCE_FILES = (
    "student_stage195_feature_residual.py", "student_stage194_residual.py",
    "scripts/train_stage195_feature_residual_pilot.py",
    "scripts/train_stage193_fresh_mixture_pilot.py",
    "scripts/preflight_stage157_teacher.py", "evaluate.py", "common.py",
    "train_experiment.py", "data/manifest.json", "data/tokenizer.json",
    "data/wikitext_train.txt", "data/wikitext_validation.txt",
)


def validate(model, data, device) -> dict:
    model.eval()
    with torch.no_grad():
        result = score(model, *data["validation"], device, "fp32", BATCH)
    result.pop("window_nll_nats")
    return result


def training_batch(tokens, generator, device):
    starts = torch.randint(len(tokens) - 257, (BATCH,), generator=generator)
    sequence = tokens[starts[:, None] + torch.arange(257)]
    return sequence[:, :256].to(device), sequence[:, 1:].to(device)


def run_arm(name, old, new, data, device, precision, run_dir, sources):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = FeatureResidualGatedLM(old).to(device)
    optimizer = torch.optim.AdamW(model.expert.parameters(), lr=PEAK_LR,
                                  betas=(.9, .999), weight_decay=.01)
    sampler = torch.Generator().manual_seed(SEED)
    arm_dir = run_dir / name
    arm_dir.mkdir()
    initial = validate(model, data, device)
    if abs(initial["bpb"] - BASE_VALIDATION_BPB) > 2e-4:
        raise ValueError(f"Unexpected zero-init validation BPB: {initial['bpb']}")
    validations = [dict(step=0, **initial)]
    history = []
    started = time.perf_counter()
    print(json.dumps({"arm": name, "validation": validations[-1]}), flush=True)
    for step in range(STEPS):
        completed = step + 1
        rate = learning_rate(step, STEPS, PEAK_LR, 100, .1, "warmup_cosine")
        for group in optimizer.param_groups:
            group["lr"] = rate
        ids, targets = training_batch(data["train"][0], sampler, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with training_autocast(device, precision):
            student_logp = model.predict_log_probs(ids).float()
            hard = F.nll_loss(student_logp.reshape(-1, 2048), targets.reshape(-1))
            if name == "teacher":
                with torch.no_grad():
                    old_logp = old.predict_log_probs(ids).float()
                    new_logp = new.predict_log_probs(ids).float()
                    teacher_probability = (torch.logaddexp(old_logp, new_logp)
                                           - math.log(2)).exp()
                distill = -(teacher_probability * student_logp).sum(-1).mean()
                loss = .5 * distill + .5 * hard
            else:
                distill, loss = None, hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss at {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.expert.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            row = dict(step=completed, loss=float(loss.detach()),
                       hard_loss=float(hard.detach()),
                       distill_loss=None if distill is None else float(distill.detach()),
                       grad_norm=grad_norm, learning_rate=rate)
            history.append(row)
            print(json.dumps({"arm": name, "train": row}), flush=True)
        if completed % 400 == 0:
            validation = validate(model, data, device)
            validations.append(dict(step=completed, **validation))
            print(json.dumps({"arm": name, "validation": validations[-1]}), flush=True)
            atomic_json_dump(dict(arm=name, history=history,
                                  validations=validations), arm_dir / "progress.json")
    checkpoint = arm_dir / "expert.pt"
    torch.save(model.expert.state_dict(), checkpoint)
    for relative, expected in sources.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Source changed during pilot: {relative}")
    result = dict(arm=name, seed=SEED, steps=STEPS, batch=BATCH,
                  primary_targets=STEPS * BATCH * 256,
                  initial_validation=validations[0],
                  best_validation=min(validations, key=lambda row: row["bpb"]),
                  final_validation=validations[-1],
                  validations=validations, history=history,
                  checkpoint_sha256=sha(checkpoint),
                  elapsed_seconds=time.perf_counter() - started,
                  no_test_scoring=True)
    atomic_json_dump(result, arm_dir / "metrics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage105", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a fresh run directory")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (not preflight.get("feasibility_gate_passed")
            or preflight.get("feature_residual_source_sha256") != PREFLIGHT_SOURCE_SHA
            or preflight.get("checkpoint_sha256") !=
               "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"):
        raise ValueError("Predeclared Stage195 feasibility gate did not pass")
    device, precision = setup("cuda", "bf16", 4)
    old, old_payload, old_module_sha = load_checkpoint(args.stage105, STAGE105_SHA, device)
    new, new_payload, new_module_sha = load_checkpoint(args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected frozen teacher implementations")
    for teacher in (old, new):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    data = load_train_validation()
    sources = {relative: sha(ROOT / relative) for relative in SOURCE_FILES}
    args.run_dir.mkdir(parents=True)
    plan = dict(protocol=PROTOCOL, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                seed=SEED, steps=STEPS, batch=BATCH, peak_lr=PEAK_LR,
                warmup_steps=100, min_lr_ratio=.1, optimizer="AdamW",
                schedule="warmup_cosine",
                optimizer_betas=[.9, .999], weight_decay=.01,
                arms=["teacher", "hard_only"], teacher_weights=[.5, .5],
                teacher_loss_weight=.5, hard_loss_weight=.5,
                validation_steps=[0, 400, 800, 1200], precision=precision,
                old_checkpoint_sha256=STAGE105_SHA, new_checkpoint_sha256=STAGE155_SHA,
                old_module_sha256=old_module_sha, new_module_sha256=new_module_sha,
                preflight_sha256=sha(args.preflight), source_hashes=sources,
                test_file_opened=False)
    atomic_json_dump(plan, args.run_dir / "run.json")
    teacher = run_arm("teacher", old, new, data, device, precision, args.run_dir, sources)
    hard = run_arm("hard_only", old, new, data, device, precision, args.run_dir, sources)
    if abs(teacher["initial_validation"]["bpb"] - hard["initial_validation"]["bpb"]) > 1e-10:
        raise ValueError("Matched arms did not start identically")
    teacher_best, hard_best = teacher["best_validation"]["bpb"], hard["best_validation"]["bpb"]
    result = dict(plan, status="completed_validation_only",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  teacher_best_bpb=teacher_best, hard_best_bpb=hard_best,
                  advance_gate_absolute=min(teacher_best, hard_best) <= 1.384686162,
                  teacher_effect_gate=(hard_best - teacher_best >= .010),
                  teacher_gain_from_base=BASE_VALIDATION_BPB - teacher_best,
                  hard_gain_from_base=BASE_VALIDATION_BPB - hard_best,
                  **device_metrics(device))
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"source_hashes": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
