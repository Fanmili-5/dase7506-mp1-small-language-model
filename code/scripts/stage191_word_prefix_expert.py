"""Run the fixed Stage191 train-only word-prefix mixture diagnostic.

This is validation-only analysis, not a deployable predictor or test score.
The entire analysis is fixed in docs/STAGE191_WORD_PREFIX_EXPERT_PLAN_20260927.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer


CODE = Path(__file__).resolve().parents[1]
PLAN = CODE / "docs/STAGE191_WORD_PREFIX_EXPERT_PLAN_20260927.md"
CACHE = CODE / "results/stage146-evidence/target-logp.npy"
EXPECTED_CHECKPOINT = (
    "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
)
EXPECTED_BASE_BPB = 1.399686162042141
WEIGHTS = (0.0, 0.05, 0.10, 0.20)
MIN_SUPPORT = 3
MAX_LETTERS = 32
HALF_WINDOWS = 736
ASCII = re.compile(r"[A-Za-z]+\Z")
START = re.compile(r"Ġ([A-Za-z]+)\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_pieces(tokenizer: Tokenizer) -> tuple[list[str | None], list[str | None]]:
    starts: list[str | None] = []
    continuations: list[str | None] = []
    for token_id in range(tokenizer.get_vocab_size()):
        spelling = tokenizer.id_to_token(token_id)
        start = START.fullmatch(spelling or "")
        starts.append(start.group(1) if start else None)
        continuations.append(spelling if spelling and ASCII.fullmatch(spelling)
                             else None)
    return starts, continuations


def next_prefix(
    previous: str | None,
    token_id: int,
    starts: list[str | None],
    continuations: list[str | None],
) -> str | None:
    if starts[token_id] is not None:
        prefix = starts[token_id]
    elif previous is not None and continuations[token_id] is not None:
        prefix = previous + continuations[token_id]
    else:
        return None
    return prefix if len(prefix) <= MAX_LETTERS else None


def train_table(
    train: list[int], starts: list[str | None], continuations: list[str | None]
) -> tuple[dict[str, dict[int, int]], int]:
    counts: dict[str, dict[int, int]] = defaultdict(dict)
    prefix = None
    observations = 0
    for current, following in zip(train[:-1], train[1:]):
        prefix = next_prefix(prefix, current, starts, continuations)
        if prefix is None:
            continue
        followers = counts[prefix]
        followers[following] = followers.get(following, 0) + 1
        observations += 1
    return counts, observations


def validation_queries(
    val: list[int], counts: dict[str, dict[int, int]],
    starts: list[str | None], continuations: list[str | None]
) -> tuple[np.ndarray, np.ndarray, dict]:
    support = {prefix: sum(followers.values()) for prefix, followers in counts.items()}
    positions: list[int] = []
    probabilities: list[float] = []
    active_by_half = [0, 0]
    correct_by_half = [0, 0]
    prefix = None
    for position, current in enumerate(val[:-1]):
        if position % 256 == 0:
            prefix = None
        prefix = next_prefix(prefix, current, starts, continuations)
        if prefix is None or prefix not in counts:
            continue
        followers = counts[prefix]
        denominator = support[prefix]
        if denominator < MIN_SUPPORT:
            continue
        target_count = followers.get(val[position + 1], 0)
        positions.append(position)
        probabilities.append(target_count / denominator)
        half = int(position // 256 >= HALF_WINDOWS)
        active_by_half[half] += 1
        correct_by_half[half] += int(target_count > 0)
    return (np.asarray(positions, dtype=np.int64),
            np.asarray(probabilities, dtype=np.float64),
            {"active_by_half": active_by_half,
             "correct_target_in_train_by_half": correct_by_half})


def analyze() -> dict:
    manifest = json.loads((CODE / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != "7506-mp1-wt2-v2":
        raise ValueError("Wrong course protocol")
    paths = {name: CODE / "data" / f"wikitext_{name}.txt"
             for name in ("train", "validation")}
    tokenizer_path = CODE / "data/tokenizer.json"
    for path in (*paths.values(), tokenizer_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Changed supplied file: {path.name}")
    cache_record = json.loads(CACHE.with_suffix(".json").read_text(encoding="utf-8"))
    if (cache_record["split"] != "validation"
            or cache_record["checkpoint_sha256"] != EXPECTED_CHECKPOINT
            or cache_record["implementation_sha256"] !=
            "1c31716eccd3bffdd2664b9ba654950a5d64d3ca0ad4e4ded817b1acbf6c9615"
            or cache_record["targets"] != 376_599
            or cache_record["utf8_bytes"] != 1_148_007
            or cache_record["array_sha256"] != sha(CACHE)):
        raise ValueError("Stage143 validation probability cache changed")
    logp = np.load(CACHE).astype(np.float64)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 2048:
        raise ValueError("Changed vocabulary")
    train = tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids
    val = tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids
    if len(logp) != len(val) - 1 or len(logp) != 376_599:
        raise ValueError("Validation target count mismatch")
    byte_count = paths["validation"].stat().st_size
    if byte_count != 1_148_007:
        raise ValueError("Validation byte count mismatch")
    baseline_bpb = -float(logp.sum()) / np.log(2) / byte_count
    if abs(baseline_bpb - EXPECTED_BASE_BPB) > 2e-5:
        raise ValueError("Stage143 target cache does not reproduce validation")

    starts, continuations = token_pieces(tokenizer)
    counts, observations = train_table(train, starts, continuations)
    positions, q_target, coverage = validation_queries(
        val, counts, starts, continuations
    )
    active_logp = logp[positions]
    q_logp = np.full(len(q_target), -np.inf, dtype=np.float64)
    positive = q_target > 0
    q_logp[positive] = np.log(q_target[positive])
    cells = []
    for weight in WEIGHTS:
        if weight == 0:
            gains = np.zeros(len(positions), dtype=np.float64)
        else:
            mixed_logp = np.logaddexp(
                np.log1p(-weight) + active_logp,
                np.log(weight) + q_logp,
            )
            gains = mixed_logp - active_logp
        half_mask = positions // 256 < HALF_WINDOWS
        half_gains = [float(gains[half_mask].sum()),
                      float(gains[~half_mask].sum())]
        full_gain = sum(half_gains)
        cells.append({
            "weight": weight,
            "gain_nats": full_gain,
            "gain_nats_by_half": half_gains,
            "gain_bpb": full_gain / np.log(2) / byte_count,
            "validation_bpb": baseline_bpb - full_gain / np.log(2) / byte_count,
        })
    primary = next(cell for cell in cells if cell["weight"] == 0.10)
    advance = (primary["gain_bpb"] >= 0.015
               and all(gain >= 2000 for gain in primary["gain_nats_by_half"]))
    return {
        "status": "diagnostic_only_not_deployable",
        "protocol": manifest["protocol"],
        "split": "validation",
        "plan_sha256": sha(PLAN),
        "source_sha256": sha(Path(__file__)),
        "target_cache_sha256": sha(CACHE),
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "train_sha256": sha(paths["train"]),
        "validation_sha256": sha(paths["validation"]),
        "tokenizer_sha256": sha(tokenizer_path),
        "train_tokens": len(train),
        "validation_targets": len(logp),
        "validation_utf8_bytes": byte_count,
        "train_prefix_observations": observations,
        "distinct_train_prefixes": len(counts),
        "distinct_train_prefix_successor_pairs": sum(map(len, counts.values())),
        "active_validation_targets": len(positions),
        "correct_target_in_train": int(np.count_nonzero(positive)),
        **coverage,
        "baseline_bpb": baseline_bpb,
        "cells": cells,
        "primary_gate_passed": bool(advance),
        "method_frozen": False,
        "test_text_opened_by_this_script": False,
        "test_scored_by_this_script": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new diagnostic output")
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
