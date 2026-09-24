"""Validation-only ceiling for Stage71/Stage76/MKN probability mixtures."""
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

from common import PROTOCOL, autocast, load_data, make_model, setup, sha, windows
from scripts.train_stage86_calibration_aware import (
    calibrated_neural_log_probs, train_log_prior)
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


PRIMARY_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
ALTERNATE_SHA = "f4c499b63db9b39ce8062b4f07eff313ec23c491952e1b4c92f22d873b23bfdf"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
ALTERNATE_WEIGHTS = tuple(index / 40 for index in range(13))  # 0..0.30
COUNT_WEIGHTS = tuple(index / 80 for index in range(13))  # 0..0.15


def target_logp(model, ids, labels, device, calibrated, log_prior=None):
    ids_device = ids.to(device)
    with autocast(device, "fp32"):
        if calibrated:
            full = calibrated_neural_log_probs(model, ids_device, log_prior)
        else:
            full = model.predict_log_probs(ids_device)
    valid = labels != -100
    target = labels.clamp_min(0).to(device).unsqueeze(-1)
    return full.gather(-1, target).squeeze(-1)[valid.to(device)].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    expected = ((args.primary, PRIMARY_SHA), (args.alternate, ALTERNATE_SHA),
                (args.counts, COUNTS_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Unexpected frozen expert checkpoint")
    primary_payload = torch.load(args.primary, map_location="cpu", weights_only=True)
    alternate_payload = torch.load(args.alternate, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (primary_payload.get("protocol") != PROTOCOL
            or primary_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or alternate_payload.get("protocol") != PROTOCOL
            or alternate_payload.get("implementation")
            != "student_hybrid_conv_structured"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected heterogeneous expert payload")

    device, _ = setup("cuda", "fp32", 4)
    primary, _ = make_model(primary_payload["implementation"],
                            primary_payload["config"], device)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    primary.load_state_dict(primary_payload["model"], strict=True); primary.eval()
    alternate.load_state_dict(alternate_payload["model"], strict=True); alternate.eval()
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior(); log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    primary_batches, alternate_batches, count_batches = [], [], []
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            primary_batches.append(target_logp(
                primary, ids, labels, device, True, log_prior))
            alternate_batches.append(target_logp(
                alternate, ids, labels, device, False))
            count_probability = count_target_probability(
                counts, ids, labels, edge_keys)[valid]
            count_batches.append(count_probability.cpu())
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    primary_probability = torch.cat(primary_batches).double().exp()
    alternate_probability = torch.cat(alternate_batches).double().exp()
    count_probability = torch.cat(count_batches).double()
    rows = []
    for alternate_weight in ALTERNATE_WEIGHTS:
        for count_weight in COUNT_WEIGHTS:
            primary_weight = 1.0 - alternate_weight - count_weight
            if primary_weight <= 0:
                continue
            probability = (primary_weight * primary_probability
                           + alternate_weight * alternate_probability
                           + count_weight * count_probability)
            nll = float(-probability.log().sum())
            rows.append(dict(
                primary_weight=primary_weight,
                alternate_weight=alternate_weight,
                count_weight=count_weight,
                nll_nats=nll, bpb=nll / math.log(2) / raw_bytes,
            ))
    best = min(rows, key=lambda row: row["bpb"])
    reference = next(row for row in rows
                     if row["alternate_weight"] == 0
                     and row["count_weight"] == .075)
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="heterogeneous_probability_ensemble_ceiling_no_export",
        primary_sha256=PRIMARY_SHA, alternate_sha256=ALTERNATE_SHA,
        counts_sha256=COUNTS_SHA,
        primary_calibration=dict(temperature=1.10,
                                 train_unigram_prior_weight=.05,
                                 copy_gate_shift=.1875),
        alternate_weights=list(ALTERNATE_WEIGHTS),
        count_weights=list(COUNT_WEIGHTS), results=rows,
        reference=reference, best=best,
        gain_over_stage85=reference["bpb"] - best["bpb"],
        train_unigram_tokens=train_tokens, targets=targets,
        utf8_bytes=raw_bytes, seconds=time.perf_counter() - started,
        validation_selected_ensemble_not_exported=True,
        deployment_exceeds_single_model_resource_budget=True,
        no_test_scoring=True,
        next_gate=("Design train-only heterogeneous distillation only if the "
                   "ensemble ceiling materially beats 1.4."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or raw_bytes != 1148007:
        raise ValueError("Coverage mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"results": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
