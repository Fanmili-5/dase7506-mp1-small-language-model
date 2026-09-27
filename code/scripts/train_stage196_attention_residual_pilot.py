"""Stage194 attention residual, retried under the course's actual resource gate.

This distinct Stage196 pilot is train/validation only. It reuses the fixed
Stage195 pilot engine with a different preregistered architecture and seed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, device_metrics, setup, sha
from scripts.preflight_stage157_teacher import STAGE105_SHA, STAGE155_SHA, load_checkpoint
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from scripts import train_stage195_feature_residual_pilot as engine
from student_stage194_residual import ResidualGatedLM
from train_experiment import atomic_json_dump

SEED = 196017
STAGE194_SOURCE_SHA = "a41b37d28867ec81fe72ff75454f8347365be458077d035f235d61ad067900ef"
SOURCE_FILES = (
    "student_stage194_residual.py", "scripts/train_stage196_attention_residual_pilot.py",
    "scripts/train_stage195_feature_residual_pilot.py",
    "scripts/train_stage193_fresh_mixture_pilot.py",
    "scripts/preflight_stage157_teacher.py", "evaluate.py", "common.py",
    "train_experiment.py", "data/manifest.json", "data/tokenizer.json",
    "data/wikitext_train.txt", "data/wikitext_validation.txt",
)


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
    if (preflight.get("checkpoint_sha256") !=
            "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
            or preflight.get("residual_source_sha256") != STAGE194_SOURCE_SHA
            or preflight.get("max_parity_logp", float("inf")) > 1e-5
            or preflight.get("max_prefix_causality_logp", float("inf")) > 1e-5
            or preflight.get("projected_assets_bytes", float("inf")) > 67_108_864):
        raise ValueError("Stage194 causal/asset evidence did not pass")
    device, precision = setup("cuda", "bf16", 4)
    old, old_payload, old_module_sha = load_checkpoint(args.stage105, STAGE105_SHA, device)
    new, new_payload, new_module_sha = load_checkpoint(args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected teacher architectures")
    for teacher in (old, new):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    data = load_train_validation()
    sources = {relative: sha(ROOT / relative) for relative in SOURCE_FILES}
    engine.FeatureResidualGatedLM = ResidualGatedLM
    engine.SEED = SEED
    engine.SOURCE_FILES = SOURCE_FILES
    args.run_dir.mkdir(parents=True)
    plan = dict(protocol=PROTOCOL, status="running", architecture="stage194_attention_residual",
                started_utc=datetime.now(timezone.utc).isoformat(),
                seed=SEED, steps=engine.STEPS, batch=engine.BATCH,
                peak_lr=engine.PEAK_LR, warmup_steps=100,
                schedule="warmup_cosine", min_lr_ratio=.1,
                optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.01,
                arms=["teacher", "hard_only"], teacher_weights=[.5, .5],
                teacher_loss_weight=.5, hard_loss_weight=.5,
                validation_steps=[0, 400, 800, 1200], precision=precision,
                old_checkpoint_sha256=STAGE105_SHA, new_checkpoint_sha256=STAGE155_SHA,
                old_module_sha256=old_module_sha, new_module_sha256=new_module_sha,
                preflight_sha256=sha(args.preflight), source_hashes=sources,
                test_file_opened=False)
    atomic_json_dump(plan, args.run_dir / "run.json")
    teacher = engine.run_arm("teacher", old, new, data, device, precision, args.run_dir, sources)
    hard = engine.run_arm("hard_only", old, new, data, device, precision, args.run_dir, sources)
    if abs(teacher["initial_validation"]["bpb"] - hard["initial_validation"]["bpb"]) > 1e-10:
        raise ValueError("Matched controls did not start identically")
    teacher_best, hard_best = teacher["best_validation"]["bpb"], hard["best_validation"]["bpb"]
    result = dict(plan, status="completed_validation_only",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  teacher_best_bpb=teacher_best, hard_best_bpb=hard_best,
                  advance_gate_absolute=min(teacher_best, hard_best) <= 1.384686162,
                  teacher_effect_gate=hard_best - teacher_best >= .010,
                  teacher_gain_from_base=engine.BASE_VALIDATION_BPB - teacher_best,
                  hard_gain_from_base=engine.BASE_VALIDATION_BPB - hard_best,
                  **device_metrics(device))
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"source_hashes": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
