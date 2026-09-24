"""Screen a frozen Stage71-to-Stage92 weight direction on full validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import (
    COUNTS_SHA, NEURAL_SHA, REFERENCE_BPB,
)
from scripts.screen_stage87_stage71_direction import interpolate
from scripts.screen_stage112_block_ablation import score
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


START_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
ALPHAS = (0.75, 0.875, 1.0, 1.125, 1.25)
EXPECTED_TARGETS = 376599
EXPECTED_BYTES = 1148007


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage137 evidence")
    if (sha(args.start) != START_SHA or sha(args.end) != NEURAL_SHA
            or sha(args.counts) != COUNTS_SHA):
        raise ValueError("Unexpected frozen checkpoint")
    start = torch.load(args.start, map_location="cpu", weights_only=True)
    end = torch.load(args.end, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (start.get("protocol") != PROTOCOL or end.get("protocol") != PROTOCOL
            or count_payload.get("protocol") != PROTOCOL
            or start.get("implementation") != "student_hybrid_conv_output_bias"
            or end.get("implementation") != start.get("implementation")
            or start.get("config") != end.get("config")
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert checkpoint payload")

    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(end["implementation"], end["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True)
    counts.eval()
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation byte count")
    edge_keys = build_target_edge_keys(counts)
    batches = []
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            batches.append((ids, labels,
                            count_target_probability(counts, ids, labels, edge_keys)))
    rows = []
    for alpha in ALPHAS:
        state = interpolate(start["model"], end["model"], alpha)
        neural.load_state_dict(state, strict=True)
        neural.eval()
        measured = score(neural, batches, log_prior, raw_bytes)
        if measured["targets"] != EXPECTED_TARGETS:
            raise ValueError("Validation coverage changed")
        if alpha == 1.0 and abs(measured["bpb"] - REFERENCE_BPB) > 2e-5:
            raise ValueError(f"Frozen Stage92 reference mismatch: {measured}")
        row = dict(alpha=alpha, **measured)
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: row["bpb"])
    evidence = dict(
        protocol=PROTOCOL, split="validation", precision="fp32",
        start_sha256=START_SHA, end_sha256=NEURAL_SHA,
        count_sha256=COUNTS_SHA, source_sha256=sha(Path(__file__)),
        alphas=list(ALPHAS), results=rows, best=best,
        stage92_reference_bpb=REFERENCE_BPB,
        improvement_over_stage92=REFERENCE_BPB - best["bpb"],
        train_unigram_tokens=train_tokens,
        no_test_scoring=True, validation_selected_weights_not_exported=True,
    )
    atomic_json_dump(evidence, args.output)
    print(json.dumps({key: value for key, value in evidence.items()
                      if key != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
