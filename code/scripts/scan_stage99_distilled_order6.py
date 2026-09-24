"""Compare train-only order-5 and order-6 MKN with calibrated Stage92."""
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
from scripts.analyze_stage98_distilled_gate import (
    NEURAL_SHA, REFERENCE_BPB, distilled_log_probs)
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


BASE_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
EXTENDED_SHA = "b901f11766c005733f769fa5582e82f586db31a41edaefe93122aae1ed471953"
WEIGHTS = tuple(i / 80 for i in range(17))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if (sha(args.neural) != NEURAL_SHA or sha(args.base) != BASE_SHA
            or sha(args.extended) != EXTENDED_SHA):
        raise ValueError("Unexpected pinned expert checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    base_payload = torch.load(args.base, map_location="cpu", weights_only=True)
    extended_payload = torch.load(args.extended, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or base_payload.get("implementation") != "student_ngram"
            or extended_payload.get("implementation") != "student_ngram"
            or extended_payload["config"].get("max_order") != 6):
        raise ValueError("Unexpected expert ancestry")

    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts = {}
    for name, payload in (("order5", base_payload), ("order6", extended_payload)):
        model, _ = make_model("student_ngram", payload["config"], device)
        model.load_state_dict(payload["model"], strict=True); model.eval()
        counts[name] = (model, build_target_edge_keys(model))
    log_prior, train_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    totals = {name: torch.zeros(len(WEIGHTS), dtype=torch.float64)
              for name in counts}
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            neural_target = distilled_log_probs(neural, ids, log_prior).gather(
                -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].double()
            for name, (model, edge_keys) in counts.items():
                count_target = count_target_probability(
                    model, ids, labels, edge_keys)[valid].double().log()
                for index, weight in enumerate(WEIGHTS):
                    mixed = neural_target if weight == 0 else torch.logaddexp(
                        neural_target + math.log1p(-weight),
                        count_target + math.log(weight))
                    totals[name][index] -= mixed.sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = {name: [dict(weight=weight, nll_nats=float(value),
                        bpb=float(value / math.log(2) / raw_bytes))
                   for weight, value in zip(WEIGHTS, values)]
            for name, values in totals.items()}
    best = {name: min(values, key=lambda row: row["bpb"])
            for name, values in rows.items()}
    base_reference = next(row for row in rows["order5"] if row["weight"] == .0625)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(base_reference["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError("Coverage or Stage94 baseline mismatch")
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="distilled_order6_fixed_mixture_diagnostic_only",
        neural_sha256=NEURAL_SHA, base_counts_sha256=BASE_SHA,
        extended_counts_sha256=EXTENDED_SHA,
        candidates=rows, best=best, base_reference=base_reference,
        order6_gain_bpb=best["order5"]["bpb"] - best["order6"]["bpb"],
        targets=targets, utf8_bytes=raw_bytes, train_unigram_tokens=train_tokens,
        seconds=time.perf_counter() - started, no_test_scoring=True,
        exported_checkpoint=False, source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
