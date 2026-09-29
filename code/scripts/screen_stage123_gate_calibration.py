"""Screen a fixed nine-point neural calibration grid under the Stage115 gate."""
from __future__ import annotations

import argparse
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

import analyze_stage50_confidence_gate as stage50
import analyze_stage60_deployable_gate as stage60
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from scripts.fit_stage100_train_gate import BASE_SHA, FEATURE_NAMES
from scripts.screen_stage110_low_cost_gate import STAGE100_SHA
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


TEMPERATURES = (1.10, 1.125, 1.15)
PRIOR_WEIGHTS = (.05, .0625, .075)
SELECTED_NAMES = (
    "neural_max_logp", "neural_margin", "highest_backoff", "highest_max_mass")
ANCHOR_WEIGHT = .0625
SLOPE_SCALE = .4
COPY_SHIFT = .25
EXPECTED_CENTER = 1.4002249460704805


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage123 output path")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA
            or sha(args.train_gate) != STAGE100_SHA):
        raise ValueError("Unexpected frozen experts or train-derived gate")
    gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    if (gate.get("protocol") != PROTOCOL
            or gate.get("no_validation_or_test_fitting") is not True
            or gate.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Unexpected gate provenance")
    indices = [FEATURE_NAMES.index(name) for name in SELECTED_NAMES]
    mean = torch.tensor([gate["feature_mean"][i] for i in indices], dtype=torch.float64)
    std = torch.tensor([gate["feature_std"][i] for i in indices], dtype=torch.float64)
    coefficients = torch.tensor([gate["coefficients"][name] for name in SELECTED_NAMES],
                                dtype=torch.float64)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 5):
        raise ValueError("Unexpected checkpoint format")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    log_prior, train_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    totals = torch.zeros((len(TEMPERATURES), len(PRIOR_WEIGHTS)), dtype=torch.float64)
    targets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            hidden = neural.features(ids).float()
            logits = neural.head(hidden) + neural.output_bias
            copy = neural.copy_distribution(hidden, ids)
            log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
            log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
            copy_gate = neural.copy_gate(hidden) + COPY_SHIFT
            log_vocab_gate = F.logsigmoid(-copy_gate)
            log_copy_gate = F.logsigmoid(copy_gate) + log_copy
            count_logp = counts.predict_log_probs(ids)
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid].double()
            _, backoff, _, max_mass, _ = stage50.count_context_features(counts, ids)
            backoff = backoff.reshape_as(ids)
            max_mass = max_mass.reshape_as(ids)
            for i, temperature in enumerate(TEMPERATURES):
                for j, prior_weight in enumerate(PRIOR_WEIGHTS):
                    vocabulary = F.log_softmax(
                        logits / temperature + prior_weight * log_prior, dim=-1)
                    neural_logp = torch.logaddexp(
                        log_vocab_gate + vocabulary, log_copy_gate)
                    top = neural_logp.topk(2, dim=-1).values
                    features = torch.stack(
                        (top[..., 0], top[..., 0] - top[..., 1],
                         backoff, max_mass), dim=-1)[valid].double()
                    slope = (((features - mean) / std) * coefficients).sum(-1)
                    weight = torch.sigmoid(
                        math.log(ANCHOR_WEIGHT / (1 - ANCHOR_WEIGHT))
                        + SLOPE_SCALE * slope)
                    weight = stage60.MIN_WEIGHT + (1 - 2 * stage60.MIN_WEIGHT) * weight
                    neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
                    totals[i, j] += stage60.mixture_nll(
                        neural_target, count_target, weight).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = [dict(temperature=temperature, prior_weight=prior_weight,
                 nll_nats=float(totals[i, j]),
                 bpb=float(totals[i, j] / math.log(2) / raw_bytes))
            for i, temperature in enumerate(TEMPERATURES)
            for j, prior_weight in enumerate(PRIOR_WEIGHTS)]
    center = next(row for row in rows if row["temperature"] == 1.125
                  and row["prior_weight"] == .0625)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(center["bpb"] - EXPECTED_CENTER) > 2e-6):
        raise ValueError(f"Stage114 center or coverage mismatch: {center}")
    result = dict(protocol=PROTOCOL, split="validation",
                  purpose="bounded_gate_aware_neural_calibration_screen",
                  neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
                  train_gate_sha256=STAGE100_SHA,
                  gate_active_features=SELECTED_NAMES, anchor_weight=ANCHOR_WEIGHT,
                  gate_slope_scale=SLOPE_SCALE, copy_gate_shift=COPY_SHIFT,
                  temperatures=TEMPERATURES, prior_weights=PRIOR_WEIGHTS,
                  candidates=rows, center=center,
                  best=min(rows, key=lambda row: row["bpb"]),
                  sub_1_4_candidates=[row for row in rows if row["bpb"] < 1.4],
                  targets=targets, utf8_bytes=raw_bytes,
                  train_unigram_tokens=train_tokens,
                  seconds=time.perf_counter() - started,
                  validation_selected_scalars_only=True,
                  no_validation_gradient_fitting=True,
                  exported_checkpoint=False, no_test_scoring=True,
                  source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": [], "sub_1_4_candidates": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
