"""Score the prespecified late average of Stage136 successor-head repair."""
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
from scripts.train_stage136_successor_head import score
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.counts) != BASE_SHA:
        raise ValueError("Wrong count checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload["protocol"] != PROTOCOL
            or neural_payload["implementation"] != "student_stage136_successor_neural"
            or neural_payload.get("seed") != 136017
            or count_payload["protocol"] != PROTOCOL):
        raise ValueError("Wrong Stage136 average or count checkpoint")
    if len(neural_payload.get("averaging_ancestry", {}).get("source_checkpoint_sha256", [])) != 4:
        raise ValueError("Expected four-checkpoint fixed average")
    device, _ = setup("cuda", "fp32", 4)
    neural, implementation_sha = make_model(neural_payload["implementation"],
                                             neural_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    neural.eval()
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    counts.load_state_dict(count_payload["model"], strict=True)
    counts.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, byte_count = load_data()["validation"]
    with torch.inference_mode():
        batches = [(ids, targets, count_target_probability(counts, ids, targets, edge_keys))
                   for ids, targets in windows(validation, 32)]
    measured = score(neural, batches, log_prior, byte_count)
    if measured["targets"] != 376599 or measured["utf8_bytes"] != 1148007:
        raise ValueError("Validation coverage mismatch")
    result = dict(protocol=PROTOCOL, split="validation", precision="fp32",
                  checkpoint_sha256=sha(args.neural),
                  count_checkpoint_sha256=BASE_SHA,
                  implementation_sha256=implementation_sha,
                  source_sha256=sha(Path(__file__)),
                  train_unigram_tokens=train_tokens,
                  fixed_average_steps=[1500, 1800, 2100, 2400],
                  no_test_scoring=True, **measured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
