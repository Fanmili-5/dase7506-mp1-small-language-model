"""Bounded feature ablation of the training-learned, anchored order-6 gate."""
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

import analyze_stage60_deployable_gate as stage60
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import (
    EXTENDED_SHA, FEATURE_COLUMNS, FEATURE_NAMES, confidence_features5)
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


STAGE100_SHA = "8608489f8c2ea6df6909ba8ff52557c024ba9dbd3c8b7032246d2579b435e991"
STAGE101_BPB = 1.4003987869888928
ANCHOR_WEIGHT = .0625
SCALES = (.05, .1, .15, .2, .3, .5)
VARIANTS = {
    "full": tuple(range(12)),
    "without_log_edges": tuple(i for i in range(12) if i != 4),
    "without_order": tuple(range(7)),
    "without_position": tuple(i for i in range(12) if i != 2),
    "without_total_mass": tuple(i for i in range(12) if i != 6),
    "neural_plus_max_mass": (0, 1, 5),
    "neural_backoff_mass": (0, 1, 3, 5),
    "clipped_one": tuple(range(12)),
    "clipped_half": tuple(range(12)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != EXTENDED_SHA
            or sha(args.train_gate) != STAGE100_SHA):
        raise ValueError("Unexpected pinned checkpoint or train-gate file")
    gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    if (gate.get("protocol") != PROTOCOL
            or gate.get("no_validation_or_test_fitting") is not True
            or gate.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Unexpected gate provenance")
    mean = torch.tensor(gate["feature_mean"], dtype=torch.float64)
    std = torch.tensor(gate["feature_std"], dtype=torch.float64)
    base_coefficients = torch.tensor(
        [gate["coefficients"][name] for name in FEATURE_NAMES],
        dtype=torch.float64)
    coefficients = []
    for label, indices in VARIANTS.items():
        value = torch.zeros_like(base_coefficients)
        source = base_coefficients
        if label == "clipped_one":
            source = source.clamp(-1., 1.)
        elif label == "clipped_half":
            source = source.clamp(-.5, .5)
        value[list(indices)] = source[list(indices)]
        coefficients.append(value)
    coefficient_matrix = torch.stack(coefficients, dim=1)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    log_prior, train_unigram_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    scale_tensor = torch.tensor(SCALES, dtype=torch.float64)
    anchor_logit = math.log(ANCHOR_WEIGHT / (1 - ANCHOR_WEIGHT))
    totals = torch.zeros((len(VARIANTS), len(SCALES)), dtype=torch.float64)
    weight_sums = torch.zeros_like(totals)
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid].double()
            features = confidence_features5(
                neural_logp, count_logp, counts, ids)[valid.flatten(), :]
            features = features[:, FEATURE_COLUMNS].double()
            slope = ((features - mean) / std) @ coefficient_matrix
            weight = torch.sigmoid(
                anchor_logit + slope[:, :, None] * scale_tensor[None, None, :])
            weight = stage60.MIN_WEIGHT + (1 - 2 * stage60.MIN_WEIGHT) * weight
            nll = stage60.mixture_nll(
                neural_target[:, None, None], count_target[:, None, None], weight)
            totals += nll.sum(0)
            weight_sums += weight.sum(0)
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = []
    for i, label in enumerate(VARIANTS):
        for j, scale in enumerate(SCALES):
            rows.append(dict(variant=label, active_features=[FEATURE_NAMES[k]
                             for k in VARIANTS[label]], slope_scale=scale,
                             mean_count_weight=float(weight_sums[i, j] / targets),
                             nll_nats=float(totals[i, j]),
                             bpb=float(totals[i, j] / math.log(2) / raw_bytes)))
    reference = next(row for row in rows
                     if row["variant"] == "full" and row["slope_scale"] == .1)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(reference["bpb"] - STAGE101_BPB) > 2e-6):
        raise ValueError("Stage101 reference or coverage mismatch")
    best = min(rows, key=lambda row: row["bpb"])
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="bounded_train_gate_feature_ablation",
        neural_sha256=NEURAL_SHA, counts_sha256=EXTENDED_SHA,
        train_gate_sha256=STAGE100_SHA, anchor_weight=ANCHOR_WEIGHT,
        variants={key: [FEATURE_NAMES[i] for i in indices]
                  for key, indices in VARIANTS.items()},
        slope_scales=SCALES, candidates=rows, reference=reference, best=best,
        gain_vs_stage101_bpb=reference["bpb"] - best["bpb"],
        train_unigram_tokens=train_unigram_tokens,
        targets=targets, utf8_bytes=raw_bytes,
        seconds=time.perf_counter() - started,
        no_validation_gradient_fitting=True, no_test_scoring=True,
        exported_checkpoint=False, source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
