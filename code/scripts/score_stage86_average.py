"""Score the fixed Stage86 neural average through frozen Stage79 calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha
from scripts.train_stage86_calibration_aware import (
    COUNTS_SHA, START_SHA, score_target_calibrated_mixture, train_log_prior)
from student_mixture_aware import build_target_edge_keys
from train_experiment import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    if sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected MKN checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert payload")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(
        neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model(
        count_payload["implementation"], count_payload["config"],
        torch.device("cpu"))
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior()
    result = score_target_calibrated_mixture(
        neural, counts, *load_data()["validation"], device, "fp32", edge_keys,
        log_prior.to(device), 32)
    result.update(
        protocol=PROTOCOL, split="validation", precision="fp32",
        purpose="fixed_stage86_parameter_average_gate",
        neural_sha256=sha(args.neural), counts_sha256=sha(args.counts),
        start_neural_sha256=START_SHA, train_unigram_tokens=train_tokens,
        no_test_scoring=True,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
