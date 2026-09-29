"""Score the frozen seven-block Stage113 parameter average with Stage94/order-six."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.screen_stage112_block_ablation import (
    COUNTS_SHA, EXPECTED_BYTES, EXPECTED_TARGETS, score)
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


AVERAGE_SHA = "856973b93225202a27b63373c5bd980a2e8513d668c25ff257576fed86c00893"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    if sha(args.neural) != AVERAGE_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected average or count checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_pruned5"
            or count_payload.get("protocol") != PROTOCOL):
        raise ValueError("Unexpected checkpoint format")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation bytes")
    edge_keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        batches = [(ids, labels, count_target_probability(counts, ids, labels, edge_keys))
                   for ids, labels in windows(validation, 32)]
    measured = score(neural, batches, log_prior, raw_bytes)
    if measured["targets"] != EXPECTED_TARGETS:
        raise ValueError("Validation coverage mismatch")
    result = dict(protocol=PROTOCOL, split="validation", purpose="stage113_average_screen",
                  neural_sha256=AVERAGE_SHA, counts_sha256=COUNTS_SHA,
                  calibration="Stage94 fixed", count_weight=.0625,
                  train_unigram_tokens=train_tokens, utf8_bytes=raw_bytes,
                  no_test_scoring=True, **measured)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
