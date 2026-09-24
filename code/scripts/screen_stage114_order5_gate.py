"""Bounded five-order MKN gate screen with frozen train-fitted coefficients."""
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
    BASE_SHA, FEATURE_COLUMNS, FEATURE_NAMES, REFERENCE, confidence_features5)
from scripts.screen_stage110_low_cost_gate import STAGE100_SHA, VARIANTS
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


SCALES = (.2, .3, .4, .5, .6, .7)
ANCHOR_WEIGHT = .0625


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage114 output path")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA
            or sha(args.train_gate) != STAGE100_SHA):
        raise ValueError("Unexpected fixed Stage114 ancestry")
    gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    if (gate.get("protocol") != PROTOCOL
            or gate.get("no_validation_or_test_fitting") is not True
            or gate.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Gate coefficients were not fit on training text")
    mean = torch.tensor(gate["feature_mean"], dtype=torch.float64)
    std = torch.tensor(gate["feature_std"], dtype=torch.float64)
    base_coeff = torch.tensor(
        [gate["coefficients"][name] for name in FEATURE_NAMES], dtype=torch.float64)
    columns = []
    for indices in VARIANTS.values():
        coefficient = torch.zeros_like(base_coeff)
        coefficient[list(indices)] = base_coeff[list(indices)]
        columns.append(coefficient)
    coefficient_matrix = torch.stack(columns, dim=1)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    log_prior, train_unigram_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    scale_tensor = torch.tensor(SCALES, dtype=torch.float64)
    anchor_logit = math.log(ANCHOR_WEIGHT / (1 - ANCHOR_WEIGHT))
    totals = torch.zeros((len(VARIANTS), len(SCALES)), dtype=torch.float64)
    fixed_total = torch.zeros((), dtype=torch.float64)
    targets = 0
    started = time.perf_counter()
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
            totals += stage60.mixture_nll(
                neural_target[:, None, None], count_target[:, None, None], weight).sum(0)
            fixed_total += stage60.mixture_nll(
                neural_target, count_target,
                torch.full_like(neural_target, ANCHOR_WEIGHT)).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = [dict(variant=name,
                 active_features=[FEATURE_NAMES[k] for k in VARIANTS[name]],
                 slope_scale=scale, nll_nats=float(totals[i, j]),
                 bpb=float(totals[i, j] / math.log(2) / raw_bytes))
            for i, name in enumerate(VARIANTS)
            for j, scale in enumerate(SCALES)]
    fixed_bpb = float(fixed_total / math.log(2) / raw_bytes)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(fixed_bpb - REFERENCE["order5"]) > 2e-5):
        raise ValueError("Order-five fixed reference or coverage mismatch")
    result = dict(
        protocol=PROTOCOL, split="validation", purpose="order5_train_gate_architecture_screen",
        neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
        train_gate_sha256=STAGE100_SHA,
        fixed_weight=ANCHOR_WEIGHT, fixed_weight_bpb=fixed_bpb,
        variants={key: [FEATURE_NAMES[i] for i in indices]
                  for key, indices in VARIANTS.items()},
        scales=SCALES, candidates=rows, best=min(rows, key=lambda row: row["bpb"]),
        sub_1_4_candidates=[row for row in rows if row["bpb"] < 1.4],
        targets=targets, utf8_bytes=raw_bytes,
        train_unigram_tokens=train_unigram_tokens,
        seconds=time.perf_counter() - started,
        no_validation_gradient_fitting=True, exported_checkpoint=False,
        no_test_scoring=True, source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": [], "sub_1_4_candidates": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
