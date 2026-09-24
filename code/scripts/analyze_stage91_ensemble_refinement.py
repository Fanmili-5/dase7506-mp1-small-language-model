"""Refine the boundary-hitting Stage90 heterogeneous ensemble grid."""
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

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage90_heterogeneous_ensemble import (
    ALTERNATE_SHA, COUNTS_SHA, PRIMARY_SHA, target_logp)
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


ALTERNATE_WEIGHTS = tuple(.25 + index * .025 for index in range(19))  # .25..70
COUNT_WEIGHTS = tuple(index / 80 for index in range(9))  # 0..10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if (sha(args.primary) != PRIMARY_SHA or sha(args.alternate) != ALTERNATE_SHA
            or sha(args.counts) != COUNTS_SHA):
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
            count_batches.append(count_target_probability(
                counts, ids, labels, edge_keys)[valid].cpu())
            targets += int(valid.sum())
    torch.cuda.synchronize(device)
    primary_probability = torch.cat(primary_batches).double().exp()
    alternate_probability = torch.cat(alternate_batches).double().exp()
    count_probability = torch.cat(count_batches).double()
    rows = []
    for alternate_weight in ALTERNATE_WEIGHTS:
        for count_weight in COUNT_WEIGHTS:
            primary_weight = 1 - alternate_weight - count_weight
            if primary_weight <= 0:
                continue
            probability = (primary_weight * primary_probability
                           + alternate_weight * alternate_probability
                           + count_weight * count_probability)
            nll = float(-probability.log().sum())
            rows.append(dict(primary_weight=primary_weight,
                             alternate_weight=alternate_weight,
                             count_weight=count_weight, nll_nats=nll,
                             bpb=nll / math.log(2) / raw_bytes))
    best = min(rows, key=lambda row: row["bpb"])
    best_neural_only = min((row for row in rows if row["count_weight"] == 0),
                           key=lambda row: row["bpb"])
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="bounded_heterogeneous_ensemble_refinement_no_export",
        primary_sha256=PRIMARY_SHA, alternate_sha256=ALTERNATE_SHA,
        counts_sha256=COUNTS_SHA,
        alternate_weights=list(ALTERNATE_WEIGHTS),
        count_weights=list(COUNT_WEIGHTS), results=rows, best=best,
        best_neural_only=best_neural_only,
        train_unigram_tokens=train_tokens, targets=targets,
        utf8_bytes=raw_bytes, seconds=time.perf_counter() - started,
        validation_selected_ensemble_not_exported=True,
        deployment_exceeds_single_model_resource_budget=True,
        no_test_scoring=True,
        next_gate=("Use the neural-only weight as a fixed train-only distillation "
                   "teacher hyperparameter; rescan MKN only after distillation."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or raw_bytes != 1148007:
        raise ValueError("Coverage mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"results": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
