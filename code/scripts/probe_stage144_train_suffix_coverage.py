"""Validation-only diagnostic of exact train-suffix continuation coverage.

This computes no inference model or score. Validation targets are used only to
measure whether a train-derived exact-match route is worth implementing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
ORDERS = (6, 7, 8, 10, 12, 16, 24, 32)
BASE = np.uint64(11400714819323198485)
PAIR_MASK = np.uint64((1 << 53) - 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(train_hist: np.ndarray, train_next: np.ndarray,
        val_hist: np.ndarray, val_next: np.ndarray, order: int) -> dict:
    ends = np.arange(order - 1, order - 1 + len(val_hist))
    valid = ends % 256 >= order - 1
    query, truth = val_hist[valid], val_next[valid]
    sorted_hist = np.sort(train_hist)
    left = np.searchsorted(sorted_hist, query, side="left")
    right = np.searchsorted(sorted_hist, query, side="right")
    multiplicity = right - left
    matched = multiplicity > 0

    # The packed key retains 53 of 64 history-hash bits, enough that accidental
    # random collisions are negligible at this corpus size; this is a screening
    # diagnostic, not a deployed lookup table or a causal correctness proof.
    train_pairs = ((train_hist & PAIR_MASK) << np.uint64(11)) | train_next
    query_pairs = ((query & PAIR_MASK) << np.uint64(11)) | truth
    sorted_pairs = np.sort(train_pairs)
    pair_left = np.searchsorted(sorted_pairs, query_pairs, side="left")
    pair_right = np.searchsorted(sorted_pairs, query_pairs, side="right")
    correct_occurrence = pair_right > pair_left
    unique = multiplicity == 1
    return {
        "order": order,
        "eligible_targets": int(len(query)),
        "matched_histories": int(np.count_nonzero(matched)),
        "matched_fraction": float(np.mean(matched)),
        "unique_histories": int(np.count_nonzero(unique)),
        "unique_correct_next": int(np.count_nonzero(unique & correct_occurrence)),
        "any_correct_next": int(np.count_nonzero(correct_occurrence)),
        "any_correct_fraction": float(np.mean(correct_occurrence)),
        "max_train_occurrences": int(multiplicity.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    paths = {split: ROOT / "data" / f"wikitext_{split}.txt"
             for split in ("train", "validation")}
    for split, path in paths.items():
        expected = manifest["sha256"][path.name]
        if sha(path) != expected:
            raise ValueError(f"Fixed {split} text changed")
    tokenizer_path = ROOT / "data/tokenizer.json"
    if sha(tokenizer_path) != manifest["sha256"][tokenizer_path.name]:
        raise ValueError("Fixed tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train = np.asarray(tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids,
                       dtype=np.uint64)
    val = np.asarray(tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids,
                     dtype=np.uint64)
    train_hist, val_hist = train + 1, val + 1
    rows = []
    for order in range(1, max(ORDERS) + 1):
        if order > 1:
            train_hist = train_hist[:-1] * BASE + train[order - 1:] + 1
            val_hist = val_hist[:-1] * BASE + val[order - 1:] + 1
        if order in ORDERS:
            result = row(train_hist[:-1], train[order:], val_hist[:-1],
                         val[order:], order)
            rows.append(result)
            print(json.dumps(result), flush=True)
    out = {
        "protocol": manifest["protocol"],
        "split": "validation",
        "purpose": "train_suffix_coverage_diagnostic_only",
        "test_scored": False,
        "train_tokens": int(len(train)),
        "validation_targets": int(len(val) - 1),
        "source_sha256": sha(Path(__file__)),
        "text_sha256": {split: sha(path) for split, path in paths.items()},
        "tokenizer_sha256": sha(tokenizer_path),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
