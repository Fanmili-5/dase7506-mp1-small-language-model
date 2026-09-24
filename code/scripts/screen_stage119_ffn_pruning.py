"""Validation loss screen for train-importance-selected SwiGLU channels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from scripts.probe_stage108_ffn_channels import sampled_windows
from scripts.screen_stage112_block_ablation import (
    COUNTS_SHA, EXPECTED_BYTES, EXPECTED_TARGETS, REFERENCE_BPB, score)
from scripts.train_stage86_calibration_aware import train_log_prior
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


RATIOS = (.10, .20, .25)
IMPORTANCE_SEED, IMPORTANCE_WINDOWS = 108017, 96


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Use pinned experts and a new output")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert format")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    data = load_data()
    train_tokens = data["train"][0]
    blocks = list(neural.blocks)
    stats = [torch.zeros(block.mlp.output.in_features, device=device)
             for block in blocks]
    importance_handles = []
    for index, block in enumerate(blocks):
        def accumulate(module, inputs, i=index):
            stats[i].add_(inputs[0].float().square().sum((0, 1)))
        importance_handles.append(block.mlp.output.register_forward_pre_hook(accumulate))
    train_sample = sampled_windows(train_tokens, IMPORTANCE_SEED, IMPORTANCE_WINDOWS)
    started = time.perf_counter()
    with torch.inference_mode():
        for chunk in train_sample.split(16):
            neural.features(chunk.to(device))
    for handle in importance_handles:
        handle.remove()
    importance = [stats[i] * block.mlp.output.weight.detach().float().square().sum(0)
                  for i, block in enumerate(blocks)]
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = data["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Unexpected validation bytes")
    edge_keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        batches = [(ids, labels, count_target_probability(counts, ids, labels, edge_keys))
                   for ids, labels in windows(validation, 32)]
    reference = score(neural, batches, log_prior, raw_bytes)
    if (reference["targets"] != EXPECTED_TARGETS
            or abs(reference["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError("Stage112 reference mismatch")
    print(json.dumps(dict(variant="reference", **reference)), flush=True)
    rows = []
    for ratio in RATIOS:
        handles = []
        removed = []
        for index, block in enumerate(blocks):
            keep = torch.ones_like(importance[index])
            lowest = importance[index].argsort()[:round(ratio * keep.numel())]
            keep[lowest] = 0
            removed.append(lowest.cpu().tolist())
            def mask(module, inputs, vector=keep):
                return (inputs[0] * vector,)
            handles.append(block.mlp.output.register_forward_pre_hook(mask))
        try:
            measured = score(neural, batches, log_prior, raw_bytes)
        finally:
            for handle in handles:
                handle.remove()
        row = dict(ratio=ratio, removed_channels_per_block=removed,
                   delta_bpb=measured["bpb"] - reference["bpb"], **measured)
        rows.append(row)
        print(json.dumps(dict(ratio=ratio, bpb=row["bpb"],
                              delta_bpb=row["delta_bpb"])), flush=True)
    result = dict(
        protocol=PROTOCOL, split="validation", purpose="ffn_channel_pruning_no_repair_screen",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        train_importance_seed=IMPORTANCE_SEED,
        train_importance_windows=IMPORTANCE_WINDOWS,
        train_unigram_tokens=train_unigram_tokens,
        reference=reference, candidates=rows,
        utf8_bytes=raw_bytes, targets=EXPECTED_TARGETS,
        seconds=time.perf_counter() - started,
        no_weight_fitting=True, exported_checkpoint=False,
        no_test_scoring=True, source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.output)
    print(json.dumps(dict(reference=reference,
                          candidates=[dict(ratio=row["ratio"], bpb=row["bpb"],
                                           delta_bpb=row["delta_bpb"])
                                      for row in rows])), flush=True)


if __name__ == "__main__":
    main()
