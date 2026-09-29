"""Screen single-block and single-FFN removals without changing model weights."""
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
from torch import nn

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


COUNTS_SHA = "b901f11766c005733f769fa5582e82f586db31a41edaefe93122aae1ed471953"
REFERENCE_BPB = 1.4012884354781747
COUNT_WEIGHT = .0625
EXPECTED_TARGETS = 376599
EXPECTED_BYTES = 1148007


class ZeroFFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def score(neural, batches, log_prior, raw_bytes):
    started = time.perf_counter()
    nll_nats = 0.0
    targets = 0
    with torch.inference_mode():
        for ids_cpu, labels_cpu, count_target_cpu in batches:
            ids = ids_cpu.to(log_prior.device)
            labels = labels_cpu.to(log_prior.device)
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_target = distilled_log_probs(neural, ids, log_prior).gather(
                -1, target).squeeze(-1)[valid]
            count_target = count_target_cpu.to(log_prior.device)[valid].clamp_min(
                torch.finfo(torch.float32).tiny).log()
            mixture = torch.logaddexp(
                neural_target + math.log1p(-COUNT_WEIGHT),
                count_target + math.log(COUNT_WEIGHT),
            )
            nll_nats -= float(mixture.double().sum())
            targets += int(valid.sum())
    if log_prior.device.type == "cuda":
        torch.cuda.synchronize(log_prior.device)
    return dict(bpb=nll_nats / math.log(2) / raw_bytes,
                nll_nats=nll_nats, targets=targets,
                neural_seconds=time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage112 output path")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen expert ancestry")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert checkpoint format")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"],
                           torch.device("cpu"))
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    if len(neural.blocks) != 8:
        raise ValueError("Expected eight Stage92 blocks")
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation byte count")
    edge_keys = build_target_edge_keys(counts)
    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            batches.append((ids, labels,
                            count_target_probability(counts, ids, labels, edge_keys)))
    print(json.dumps(dict(count_batches=len(batches),
                          count_precompute_seconds=time.perf_counter() - started)),
          flush=True)
    reference = score(neural, batches, log_prior, raw_bytes)
    if (reference["targets"] != EXPECTED_TARGETS
            or abs(reference["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError(f"Frozen reference mismatch: {reference}")
    print(json.dumps(dict(variant="reference", **reference)), flush=True)
    rows = []
    for kind in ("whole_block", "ffn_only"):
        for index in range(len(neural.blocks)):
            if kind == "whole_block":
                original = neural.blocks[index]
                neural.blocks[index] = nn.Identity()
            else:
                original = neural.blocks[index].mlp
                neural.blocks[index].mlp = ZeroFFN()
            try:
                measured = score(neural, batches, log_prior, raw_bytes)
            finally:
                if kind == "whole_block":
                    neural.blocks[index] = original
                else:
                    neural.blocks[index].mlp = original
            row = dict(variant=kind, zero_based_block=index,
                       one_based_block=index + 1, **measured,
                       delta_bpb=measured["bpb"] - reference["bpb"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    if any(row["targets"] != EXPECTED_TARGETS for row in rows):
        raise ValueError("Ablation coverage mismatch")
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="single_structural_ablation_diagnostic_only",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        calibration="Stage94 fixed", count_weight=COUNT_WEIGHT,
        reference=reference, candidates=rows,
        best_whole_block=min((r for r in rows if r["variant"] == "whole_block"),
                             key=lambda r: r["bpb"]),
        best_ffn_only=min((r for r in rows if r["variant"] == "ffn_only"),
                          key=lambda r: r["bpb"]),
        train_unigram_tokens=train_unigram_tokens,
        targets=EXPECTED_TARGETS, utf8_bytes=raw_bytes,
        no_weight_fitting=True, exported_checkpoint=False,
        no_test_scoring=True, source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(dict(best_whole_block=result["best_whole_block"],
                          best_ffn_only=result["best_ffn_only"])), flush=True)


if __name__ == "__main__":
    main()
