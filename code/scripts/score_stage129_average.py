"""Score the fixed four-checkpoint merged Stage129 neural average."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.screen_stage112_block_ablation import EXPECTED_BYTES, EXPECTED_TARGETS, score
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage129 average output")
    if sha(args.counts) != BASE_SHA:
        raise ValueError("Unexpected five-order count checkpoint")
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    counts_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_output_bias"
            or payload.get("seed") != 129017
            or payload.get("averaging_ancestry", {}).get("source_count") != 4
            or counts_payload.get("protocol") != PROTOCOL
            or counts_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected Stage129 average ancestry")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(payload["implementation"], payload["config"], device)
    counts, _ = make_model("student_ngram", counts_payload["config"], torch.device("cpu"))
    neural.load_state_dict(payload["model"], strict=True)
    counts.load_state_dict(counts_payload["model"], strict=True)
    neural.eval(); counts.eval()
    log_prior, train_tokens = train_log_prior()
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Validation byte count changed")
    edge_keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        batches = [
            (ids, labels, count_target_probability(counts, ids, labels, edge_keys))
            for ids, labels in windows(validation, 32)]
    measured = score(neural, batches, log_prior.to(device), raw_bytes)
    if measured["targets"] != EXPECTED_TARGETS:
        raise ValueError("Stage129 average target coverage mismatch")
    result = dict(protocol=PROTOCOL, split="validation", precision="fp32",
                  purpose="stage129_fixed_four_merged_checkpoint_average",
                  neural_sha256=sha(args.neural), counts_sha256=BASE_SHA,
                  calibration="Stage94 fixed", count_weight=.0625,
                  train_unigram_tokens=train_tokens, utf8_bytes=raw_bytes,
                  no_test_scoring=True, **measured)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
