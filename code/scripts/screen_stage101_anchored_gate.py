"""Select a bounded validation slope for a gate learned on training text."""
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
    BASE_SHA, EXTENDED_SHA, FEATURE_COLUMNS, FEATURE_NAMES, REFERENCE,
    confidence_features5)
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


STAGE100_SHA = "8608489f8c2ea6df6909ba8ff52557c024ba9dbd3c8b7032246d2579b435e991"
SCALES = (0., .05, .1, .15, .2, .3, .4, .5, .75, 1.)
ANCHOR_WEIGHT = .0625


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if (sha(args.neural) != NEURAL_SHA or sha(args.base) != BASE_SHA
            or sha(args.extended) != EXTENDED_SHA
            or sha(args.train_gate) != STAGE100_SHA):
        raise ValueError("Unexpected pinned expert or train-gate file")
    gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    if (gate.get("protocol") != PROTOCOL
            or gate.get("no_validation_or_test_fitting") is not True
            or gate.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Unexpected gate training provenance")
    mean = torch.tensor(gate["feature_mean"], dtype=torch.float64)
    std = torch.tensor(gate["feature_std"], dtype=torch.float64)
    coefficients = torch.tensor(
        [gate["coefficients"][name] for name in FEATURE_NAMES],
        dtype=torch.float64)
    if not all(torch.isfinite(value).all() for value in (mean, std, coefficients)):
        raise ValueError("Non-finite gate statistics")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payloads = {
        "order5": torch.load(args.base, map_location="cpu", weights_only=True),
        "order6": torch.load(args.extended, map_location="cpu", weights_only=True),
    }
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts = {}
    for label, payload in count_payloads.items():
        model, _ = make_model("student_ngram", payload["config"], device)
        model.load_state_dict(payload["model"], strict=True); model.eval()
        counts[label] = model
    log_prior, train_unigram_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    scale_tensor = torch.tensor(SCALES, dtype=torch.float64)
    anchor_logit = math.log(ANCHOR_WEIGHT / (1 - ANCHOR_WEIGHT))
    totals = {label: torch.zeros(len(SCALES), dtype=torch.float64)
              for label in counts}
    weight_sums = {label: torch.zeros(len(SCALES), dtype=torch.float64)
                   for label in counts}
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
            for name, model in counts.items():
                count_logp = model.predict_log_probs(ids)
                count_target = count_logp.gather(-1, target).squeeze(-1)[valid].double()
                features = confidence_features5(
                    neural_logp, count_logp, model, ids)[valid.flatten(), :]
                features = features[:, FEATURE_COLUMNS].double()
                slope = ((features - mean) / std) @ coefficients
                weight = torch.sigmoid(anchor_logit + slope[:, None] * scale_tensor)
                weight = (stage60.MIN_WEIGHT
                          + (1 - 2 * stage60.MIN_WEIGHT) * weight)
                nll = stage60.mixture_nll(
                    neural_target[:, None], count_target[:, None], weight)
                totals[name] += nll.sum(0)
                weight_sums[name] += weight.sum(0)
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = {name: [dict(slope_scale=scale, mean_count_weight=float(weight_sum / targets),
                        nll_nats=float(nll),
                        bpb=float(nll / math.log(2) / raw_bytes))
                   for scale, weight_sum, nll in zip(SCALES, weight_sums[name], totals[name])]
            for name in counts}
    best = {name: min(values, key=lambda row: row["bpb"])
            for name, values in rows.items()}
    if (targets != 376599 or raw_bytes != 1148007
            or any(abs(rows[name][0]["bpb"] - REFERENCE[name]) > 2e-5
                   for name in counts)):
        raise ValueError("Coverage or fixed-anchor baseline mismatch")
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="bounded_selection_of_train_learned_gate_slope",
        neural_sha256=NEURAL_SHA, base_counts_sha256=BASE_SHA,
        extended_counts_sha256=EXTENDED_SHA,
        train_gate_sha256=STAGE100_SHA, anchor_weight=ANCHOR_WEIGHT,
        slope_scales=SCALES, candidates=rows, best=best,
        train_gate_feature_names=FEATURE_NAMES,
        train_unigram_tokens=train_unigram_tokens,
        targets=targets, utf8_bytes=raw_bytes,
        seconds=time.perf_counter() - started,
        validation_selects_only_discrete_slope=True,
        no_validation_gradient_fitting=True, no_test_scoring=True,
        exported_checkpoint=False, source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
