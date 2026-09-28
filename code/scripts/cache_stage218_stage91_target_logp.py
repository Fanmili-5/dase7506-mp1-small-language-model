"""Cache Stage91 frozen-mixture validation target log-probs; no test access."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from common import PROTOCOL, make_model, setup, sha, windows
from scripts.analyze_stage90_heterogeneous_ensemble import (
    PRIMARY_SHA, ALTERNATE_SHA, COUNTS_SHA, target_logp,
)
from scripts.train_stage86_calibration_aware import train_log_prior
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from student_mixture_aware import build_target_edge_keys, count_target_probability


EXPECTED_STAGE91_BPB = 1.3816233893
WEIGHTS = (0.5375, 0.4250, 0.0375)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--alternate", required=True, type=Path)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose a new Stage218 cache path")
    for path, expected in ((args.primary, PRIMARY_SHA),
                           (args.alternate, ALTERNATE_SHA),
                           (args.counts, COUNTS_SHA)):
        if sha(path) != expected:
            raise ValueError("Frozen expert checkpoint changed: " + str(path))
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path in (args.primary, args.alternate, args.counts)]
    expected_kinds = ("student_hybrid_conv_output_bias",
                      "student_hybrid_conv_structured", "student_ngram")
    for payload, kind in zip(payloads, expected_kinds):
        if payload.get("protocol") != PROTOCOL or payload.get("implementation") != kind:
            raise ValueError("Unexpected frozen expert payload")
    device, _ = setup("cuda", "fp32", 4)
    primary, _ = make_model(payloads[0]["implementation"], payloads[0]["config"], device)
    alternate, _ = make_model(payloads[1]["implementation"], payloads[1]["config"], device)
    counts, _ = make_model(payloads[2]["implementation"], payloads[2]["config"],
                           torch.device("cpu"))
    for model, payload in zip((primary, alternate, counts), payloads):
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
    edge_keys = build_target_edge_keys(counts)
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_train_validation()["validation"]
    chunks = []
    started = time.perf_counter()
    with torch.inference_mode():
        for ids, labels in windows(validation, 32):
            valid = labels != -100
            primary_logp = target_logp(primary, ids, labels, device, True, log_prior)
            alternate_logp = target_logp(alternate, ids, labels, device, False)
            count_probability = count_target_probability(
                counts, ids, labels, edge_keys)[valid].cpu().double()
            probability = (WEIGHTS[0] * primary_logp.double().exp()
                           + WEIGHTS[1] * alternate_logp.double().exp()
                           + WEIGHTS[2] * count_probability)
            if not torch.isfinite(probability).all() or (probability <= 0).any():
                raise ValueError("Invalid Stage91 target probability")
            chunks.append(probability.log().numpy())
    torch.cuda.synchronize(device)
    logp = np.concatenate(chunks).astype(np.float64)
    bpb = -float(logp.sum()) / math.log(2) / raw_bytes
    if len(logp) != 376_599 or raw_bytes != 1_148_007:
        raise ValueError("Incomplete validation coverage")
    if abs(bpb - EXPECTED_STAGE91_BPB) > 2e-5:
        raise ValueError(f"Frozen Stage91 reproduction mismatch: {bpb}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logp)
    meta = {
        "purpose": "frozen_expert_validation_target_probability_diagnostic_only",
        "protocol": PROTOCOL, "split": "validation", "no_test_scoring": True,
        "targets": len(logp), "utf8_bytes": raw_bytes, "bpb": bpb,
        "fixed_weights": list(WEIGHTS), "train_unigram_tokens": train_tokens,
        "primary_sha256": PRIMARY_SHA, "alternate_sha256": ALTERNATE_SHA,
        "counts_sha256": COUNTS_SHA,
        "source_sha256": sha(Path(__file__)),
        "dependencies_sha256": {
            name: sha(ROOT / name) for name in (
                "scripts/analyze_stage90_heterogeneous_ensemble.py",
                "scripts/train_stage86_calibration_aware.py",
                "scripts/train_stage193_fresh_mixture_pilot.py",
                "student_mixture_aware.py", "common.py",
                "data/manifest.json", "data/tokenizer.json",
                "data/wikitext_train.txt", "data/wikitext_validation.txt",
            )
        },
        "array_sha256": sha(args.output),
        "seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
