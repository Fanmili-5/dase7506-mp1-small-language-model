"""Score preregistered contiguous Stage71 checkpoint soups on validation."""
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
    COUNTS_SHA, score_target_calibrated_mixture, train_log_prior)
from student_mixture_aware import build_target_edge_keys
from train_experiment import atomic_json_dump


CHECKPOINT_SHAS = {
    "step-002400.pt": "126818d4c2624e91dec7fa04cf9e1ce4cb8d927f7bcea32a17091c162d4dc236",
    "step-002700.pt": "2a331d80d8bc839e0da66fa1eb6ed1653b46e496a88af19f7bfe32cad6aad6bb",
    "step-003000.pt": "ca6f9fcaec3dc315ee359dc2d356cb4bd249133428c8e784581cf5c1fde980e8",
    "step-003300.pt": "60c8d1aa3877c8bac848ab96f9909ed920c63af2599eaf835636b437ee604f01",
    "step-003600.pt": "55a1851d39b4bbf467373b8ea806fbc17d86837187d248a83a1a76fb75f136a6",
}


def average_states(states: list[dict]) -> dict:
    if not states:
        raise ValueError("At least one state is required")
    result = {}
    for name, first in states[0].items():
        tensors = [state[name] for state in states]
        if any(tensor.shape != first.shape or tensor.dtype != first.dtype
               for tensor in tensors):
            raise ValueError("Checkpoint tensor metadata differs: " + name)
        if first.is_floating_point():
            value = first.to(torch.float64)
            for tensor in tensors[1:]:
                value.add_(tensor.to(torch.float64))
            result[name] = value.div_(len(tensors)).to(first.dtype)
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError("Non-floating state differs: " + name)
            result[name] = first
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    if sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected MKN checkpoint")

    paths = [args.checkpoint_dir / name for name in CHECKPOINT_SHAS]
    for path in paths:
        if sha(path) != CHECKPOINT_SHAS[path.name]:
            raise ValueError("Unexpected checkpoint: " + path.name)
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path in paths]
    reference = payloads[0]
    for payload in payloads:
        if (payload.get("protocol") != PROTOCOL
                or payload.get("implementation")
                != "student_hybrid_conv_output_bias"
                or payload.get("config") != reference.get("config")):
            raise ValueError("Unexpected Stage71 checkpoint payload")

    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected count payload")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(reference["implementation"], reference["config"], device)
    counts, _ = make_model(
        count_payload["implementation"], count_payload["config"],
        torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior(); log_prior = log_prior.to(device)
    validation = load_data()["validation"]

    states = [payload["model"] for payload in payloads]
    rows = []
    # Complete preregistered space: all contiguous intervals among the five
    # fixed, equally spaced late checkpoints (15 candidates total).
    for start in range(len(states)):
        for stop in range(start + 1, len(states) + 1):
            neural.load_state_dict(average_states(states[start:stop]), strict=True)
            neural.eval()
            score = score_target_calibrated_mixture(
                neural, counts, *validation, device, "fp32", edge_keys,
                log_prior, 32)
            row = dict(
                start_step=2400 + 300 * start,
                end_step=2400 + 300 * (stop - 1),
                checkpoint_count=stop - start,
                **score,
            )
            rows.append(row); print(json.dumps(row), flush=True)

    best = min(rows, key=lambda row: row["bpb"])
    output = dict(
        protocol=PROTOCOL, split="validation",
        purpose="preregistered_contiguous_checkpoint_soup_screen",
        checkpoint_sha256=CHECKPOINT_SHAS, counts_sha256=COUNTS_SHA,
        candidate_count=len(rows), results=rows, best=best,
        stage85_reference_bpb=1.4030241331845539,
        gain_over_stage85=1.4030241331845539 - best["bpb"],
        train_unigram_tokens=train_tokens,
        validation_selected_weights_not_exported=True, no_test_scoring=True,
        next_gate=("Export and run the exact fused resource qualification only "
                   "if the best contiguous soup materially improves Stage85."),
        source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(output, args.output)
    print(json.dumps(output | {"results": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
