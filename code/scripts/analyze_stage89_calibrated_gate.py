"""Cross-fit causal confidence gates for the calibrated Stage85 experts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import analyze_stage50_confidence_gate as stage50
import analyze_stage60_deployable_gate as stage60
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.train_stage86_calibration_aware import (
    calibrated_neural_log_probs, train_log_prior)
from train_experiment import atomic_json_dump


NEURAL_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen Stage71 or Stage25 checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected Stage71 neural and Stage25 MKN experts")

    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    log_prior, train_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    feature_batches, neural_batches, count_batches, half_batches = [], [], [], []
    totals = torch.zeros(len(stage60.WEIGHTS), dtype=torch.float64)
    targets = 0; started = time.perf_counter()
    number_of_windows = math.ceil((len(validation) - 1) / 256)
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = calibrated_neural_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            all_features = stage50.confidence_features(
                neural_logp, count_logp, counts, ids)[valid.flatten()]
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid]
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid]
            feature_batches.append(all_features.cpu())
            neural_batches.append(neural_target.cpu())
            count_batches.append(count_target.cpu())
            indices = torch.arange(batch * 32, batch * 32 + ids.shape[0])
            halves = (indices >= number_of_windows // 2).unsqueeze(1).expand_as(valid)
            half_batches.append(halves[valid].cpu())
            for index, scalar in enumerate(stage60.WEIGHTS):
                if scalar == 0:
                    totals[index] -= neural_target.double().sum()
                else:
                    weight = torch.full_like(neural_target, scalar).double()
                    totals[index] += stage60.mixture_nll(
                        neural_target.double(), count_target.double(), weight).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)

    all_features = torch.cat(feature_batches)
    neural_target = torch.cat(neural_batches)
    count_target = torch.cat(count_batches)
    second_half = torch.cat(half_batches).bool(); first_half = ~second_half
    feature_results = {}
    for label, columns in stage60.FEATURE_SETS.items():
        names = tuple(stage50.FEATURE_NAMES[index] for index in columns)
        features = all_features[:, columns]
        folds = [
            dict(train="first_half", evaluate="second_half",
                 **stage60.fit_gate(features, names, neural_target, count_target,
                                    first_half, second_half)),
            dict(train="second_half", evaluate="first_half",
                 **stage60.fit_gate(features, names, neural_target, count_target,
                                    second_half, first_half)),
        ]
        nll = sum(fold["nll_nats"] for fold in folds)
        feature_results[label] = dict(
            feature_names=list(names), folds=folds,
            crossfit_bpb=nll / math.log(2) / raw_bytes,
            deployability=("low_overhead_candidate"
                           if label != "full_diagnostic_ceiling"
                           else "diagnostic_ceiling_requires_dense_count_statistics"),
        )
    rows = [dict(weight=weight, nll_nats=float(total),
                 bpb=float(total / math.log(2) / raw_bytes))
            for weight, total in zip(stage60.WEIGHTS, totals)]
    best_fixed = min(rows, key=lambda row: row["bpb"])
    best_low_overhead = min(
        (value for key, value in feature_results.items()
         if key != "full_diagnostic_ceiling"),
        key=lambda row: row["crossfit_bpb"])
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="calibrated_gate_crossfit_diagnostic_only_no_export",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        calibration=dict(temperature=1.10, train_unigram_prior_weight=.05,
                         copy_gate_shift=.1875),
        fixed_weight_grid=rows, best_fixed=best_fixed,
        feature_sets=feature_results, best_low_overhead=best_low_overhead,
        low_overhead_gain=best_fixed["bpb"] - best_low_overhead["crossfit_bpb"],
        train_unigram_tokens=train_tokens, targets=targets, utf8_bytes=raw_bytes,
        seconds=time.perf_counter() - started,
        validation_fit_diagnostic_only=True, exported_checkpoint=False,
        no_test_scoring=True,
        next_gate=("Train a proxy neural and temporary MKN excluding a fixed "
                   "train calibration slice only if the low-overhead cross-fit "
                   "result materially beats 1.4."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or raw_bytes != 1148007 or all_features.shape[1] != 19:
        raise ValueError("Coverage or feature shape mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
