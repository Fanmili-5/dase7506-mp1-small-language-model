"""Screen bounded interpolation along the Stage67-to-Stage71 weight direction."""
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


START_SHA = "3be9468b122da4c486726e8bcc59692005d0fc90dcbdd6a59c41b24d42d20b0f"
END_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
ALPHAS = tuple(.5 + .125 * index for index in range(9))


def interpolate(start: dict, end: dict, alpha: float) -> dict:
    result = {}
    if start.keys() != end.keys():
        raise ValueError("Checkpoint state keys differ")
    for name, first in start.items():
        second = end[name]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("Checkpoint tensor metadata differs: " + name)
        if first.is_floating_point():
            result[name] = torch.lerp(first, second, alpha)
        else:
            if not torch.equal(first, second):
                raise ValueError("Non-floating state differs: " + name)
            result[name] = first
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    if (sha(args.start) != START_SHA or sha(args.end) != END_SHA
            or sha(args.counts) != COUNTS_SHA):
        raise ValueError("Unexpected frozen checkpoint")
    start = torch.load(args.start, map_location="cpu", weights_only=True)
    end = torch.load(args.end, map_location="cpu", weights_only=True)
    counts_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (start.get("protocol") != PROTOCOL or end.get("protocol") != PROTOCOL
            or start.get("implementation") != "student_hybrid_conv_output_bias"
            or end.get("implementation") != start.get("implementation")
            or start.get("config") != end.get("config")
            or counts_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert payload")

    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(end["implementation"], end["config"], device)
    counts, _ = make_model(
        counts_payload["implementation"], counts_payload["config"],
        torch.device("cpu"))
    counts.load_state_dict(counts_payload["model"], strict=True); counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior(); log_prior = log_prior.to(device)
    validation = load_data()["validation"]
    rows = []
    for alpha in ALPHAS:
        state = interpolate(start["model"], end["model"], alpha)
        neural.load_state_dict(state, strict=True); neural.eval()
        score = score_target_calibrated_mixture(
            neural, counts, *validation, device, "fp32", edge_keys,
            log_prior, 32)
        row = dict(alpha=alpha, **score)
        rows.append(row); print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: row["bpb"])
    output = dict(
        protocol=PROTOCOL, split="validation", purpose="bounded_weight_direction_screen",
        start_sha256=START_SHA, end_sha256=END_SHA, counts_sha256=COUNTS_SHA,
        alphas=list(ALPHAS), results=rows, best=best,
        stage85_reference_bpb=1.4030241331845539,
        gain_over_stage85=1.4030241331845539 - best["bpb"],
        train_unigram_tokens=train_tokens,
        validation_selected_weights_not_exported=True, no_test_scoring=True,
        next_gate=("Refine and export only if the bounded direction improves "
                   "Stage85 materially."),
        source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(output, args.output)
    print(json.dumps(output | {"results": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
