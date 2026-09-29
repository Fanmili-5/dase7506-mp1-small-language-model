"""Train only a zero-start morphology residual over frozen Stage105 outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from student_stage160_morphology import (FrozenBaseWithMorphology,
                                         MorphologyResidual, TOKENIZER_SHA256)
from train_experiment import atomic_json_dump, atomic_torch_save, learning_rate

BASE_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
STAGE143_BPB = 1.399686162042141
SEED = 160017
STEPS = 2400
BATCH = 16
PEAK_LR = 1e-3
SOURCE_FILES = (
    "student_stage160_morphology.py", "student_stage105_gated_singlepass.py",
    "student_stage104_gated_fast.py", "student_stage103_gated.py",
    "student_ngram_collapsed.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py",
    "student.py", "student_ngram.py", "common.py", "evaluate.py",
    "train_experiment.py", "scripts/train_stage160_morphology_pilot.py",
    "data/manifest.json", "data/tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new pilot directory")
    if sha(args.base) != BASE_SHA:
        raise ValueError("Unexpected frozen Stage105 base")
    payload = torch.load(args.base, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage105_gated_singlepass"):
        raise ValueError("Unexpected frozen base implementation")
    device, precision = setup("cuda", "fp32", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    base, base_module_sha = make_model(payload["implementation"],
                                       payload["config"], device)
    base.load_state_dict(payload["model"], strict=True)
    base.eval()
    residual = MorphologyResidual().to(device)
    wrapper = FrozenBaseWithMorphology(base, residual).eval()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=PEAK_LR,
                                  betas=(.9, .999), weight_decay=.01)
    data = load_data()
    tokens = data["train"][0]
    sampler = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    args.run_dir.mkdir(parents=True)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        physical_batch=BATCH, effective_batch=BATCH, context=256,
        primary_targets=STEPS * BATCH * 256,
        frozen_base_sha256=BASE_SHA, frozen_base_module_sha256=base_module_sha,
        morphology_module_sha256=sha(ROOT / "student_stage160_morphology.py"),
        tokenizer_sha256=TOKENIZER_SHA256,
        residual_parameters=sum(p.numel() for p in residual.parameters()),
        base_parameters_trainable=0,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.01,
        peak_learning_rate=PEAK_LR, warmup_steps=50,
        minimum_learning_rate=PEAK_LR * .1,
        learning_rate_schedule="warmup_cosine", precision=precision,
        source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(),
        validation_labels_not_used_for_gradients=True,
        training_uses_supplied_train_text_only=True, no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    started = time.perf_counter(); validation_seconds = 0.0
    before = time.perf_counter()
    initial = score(wrapper, *data["validation"], device, "fp32", 32)
    initial.pop("window_nll_nats")
    validation_seconds += time.perf_counter() - before
    if abs(initial["bpb"] - STAGE143_BPB) > 2e-5:
        raise ValueError(f"Zero-start morphology disagrees with Stage143: {initial['bpb']}")
    validations = [dict(step=0, **initial)]
    history = []
    print(json.dumps({"validation": validations[-1]}), flush=True)
    for step in range(STEPS):
        completed = step + 1
        rate = learning_rate(step, STEPS, PEAK_LR, 50, .1, "warmup_cosine")
        for group in optimizer.param_groups:
            group["lr"] = rate
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=sampler)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids, targets = sequence[:, :256].to(device), sequence[:, 1:].to(device)
        optimizer.zero_grad(set_to_none=True)
        residual.train()
        logp = wrapper.predict_log_probs(ids)
        loss = F.nll_loss(logp.reshape(-1, 2048), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage160 loss at {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       grad_norm=grad_norm, learning_rate=rate,
                       primary_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            residual.eval()
            before = time.perf_counter()
            result = score(wrapper, *data["validation"], device, "fp32", 32)
            result.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            validations.append(dict(step=completed, **result))
            print(json.dumps({"validation": validations[-1]}), flush=True)
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    checkpoint = args.run_dir / "residual.pt"
    atomic_torch_save(dict(
        protocol=PROTOCOL, implementation="student_stage160_morphology",
        config=dict(width=288, residual_width=128,
                    tokenizer_sha256=TOKENIZER_SHA256),
        model=residual.state_dict(), frozen_base_sha256=BASE_SHA,
        seed=SEED, train_tokens=STEPS * BATCH * 256,
        no_test_scoring=True), checkpoint)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during Stage160 pilot: {name}")
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        history=history, validation_history=validations,
        final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        endpoint_gain_vs_stage143_bpb=STAGE143_BPB - validations[-1]["bpb"],
        pilot_gate_passed=STAGE143_BPB - validations[-1]["bpb"] >= .005,
        checkpoint_sha256=sha(checkpoint), **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
