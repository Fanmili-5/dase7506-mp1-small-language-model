"""Score fixed balanced-student average with the unchanged Stage94 order-five mixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import COUNTS_SHA, REFERENCE_BPB
from scripts.screen_stage112_block_ablation import score
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
        parser.error("Refusing to overwrite Stage140 average evidence")
    if sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected count checkpoint")
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_output_bias"
            or config.get("width") != 300 or config.get("depth") != 8
            or config.get("conv_layers") != [2, 3, 5, 6, 8]
            or config.get("output_bias") is not True
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected Stage140 average payload")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(payload["implementation"], config, device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    neural.load_state_dict(payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != 1148007:
        raise ValueError("Validation byte count changed")
    edge_keys = build_target_edge_keys(counts)
    batches = []
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            batches.append((ids, labels,
                            count_target_probability(counts, ids, labels, edge_keys)))
    measured = score(neural, batches, log_prior, raw_bytes)
    if measured["targets"] != 376599:
        raise ValueError("Incomplete validation target coverage")
    result = dict(
        protocol=PROTOCOL, split="validation", precision="fp32",
        purpose="fixed_stage140_average_stage94_order5_gate",
        neural_sha256=sha(args.neural), counts_sha256=COUNTS_SHA,
        source_sha256=sha(Path(__file__)),
        train_unigram_tokens=train_tokens,
        stage92_fixed_reference_bpb=REFERENCE_BPB,
        next_gate_requires_bpb_below_1_4=measured["bpb"] < 1.4,
        no_test_scoring=True, **measured,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
