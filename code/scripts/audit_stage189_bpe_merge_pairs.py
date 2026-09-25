"""Train/validation-only incidence audit for fixed tokenizer merge pairs.

This first gate never loads the test split or a language-model checkpoint.
It does not change the tokenizer, model, or any inference asset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROTOCOL = "7506-mp1-wt2-v2"
EXPECTED_TOKENIZER_SHA256 = (
    "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e"
)
EXPECTED_MERGES = 1792
EXPECTED_VOCAB = 2048


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_merge_pairs(tokenizer_json: dict) -> set[tuple[int, int]]:
    model = tokenizer_json["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    if (tokenizer_json["pre_tokenizer"]["type"] != "ByteLevel"
            or model["type"] != "BPE" or len(vocab) != EXPECTED_VOCAB
            or len(merges) != EXPECTED_MERGES):
        raise ValueError("Unexpected fixed BPE tokenizer structure")
    if set(vocab.values()) != set(range(EXPECTED_VOCAB)):
        raise ValueError("Vocab IDs are not 0..2047")
    pairs = set()
    for merge in merges:
        if not isinstance(merge, list) or len(merge) != 2:
            raise ValueError("Expected two-symbol merge arrays")
        left, right = merge
        if left not in vocab or right not in vocab or left + right not in vocab:
            raise ValueError("Merge references missing vocabulary symbol")
        pair = (vocab[left], vocab[right])
        if pair in pairs:
            raise ValueError("Duplicate direct merge pair")
        pairs.add(pair)
    if len(pairs) != EXPECTED_MERGES:
        raise ValueError("Incorrect unique merge count")
    return pairs


def audit_ids(ids: list[int], merge_pairs: set[tuple[int, int]]) -> dict:
    incidence: Counter[tuple[int, int]] = Counter()
    window_incidence: Counter[tuple[int, int]] = Counter()
    for position, pair in enumerate(zip(ids, ids[1:])):
        if pair in merge_pairs:
            incidence[pair] += 1
            # The scorer's independent windows cover x[i] -> y[i+1] for
            # every i; even the last position in a window has a target.
            window_start = position // 256 * 256
            if position < min(window_start + 256, len(ids) - 1):
                window_incidence[pair] += 1
    if incidence != window_incidence:
        raise ValueError("Independent-window target accounting mismatch")
    return {
        "tokens": len(ids),
        "scored_adjacent_targets": max(0, len(ids) - 1),
        "independent_windows": (max(0, len(ids) - 1) + 255) // 256,
        "masked_true_pair_occurrences": sum(incidence.values()),
        "distinct_masked_true_pairs": len(incidence),
        "masked_true_pairs_top20": [
            {"previous_id": left, "target_id": right, "occurrences": count}
            for (left, right), count in incidence.most_common(20)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unexpected course protocol")
    tokenizer_path = DATA / "tokenizer.json"
    if (sha(tokenizer_path) != EXPECTED_TOKENIZER_SHA256
            or manifest["sha256"]["tokenizer.json"] != EXPECTED_TOKENIZER_SHA256):
        raise ValueError("Fixed tokenizer hash changed")
    tokenizer_json = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    merge_pairs = direct_merge_pairs(tokenizer_json)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    splits = {}
    for split in ("train", "validation"):
        name = f"wikitext_{split}.txt"
        path = DATA / name
        if sha(path) != manifest["sha256"][name]:
            raise ValueError(f"Fixed {split} text hash changed")
        raw = path.read_bytes()
        ids = tokenizer.encode(raw.decode("utf-8")).ids
        splits[split] = {"utf8_bytes": len(raw), "text_sha256": sha(path),
                         **audit_ids(ids, merge_pairs)}
        print(f"{split}: {splits[split]['masked_true_pair_occurrences']} masked true pairs", flush=True)
    result = {
        "protocol": PROTOCOL,
        "purpose": "stage189_direct_merge_pair_incidence_only",
        "source_sha256": sha(Path(__file__)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "vocab": EXPECTED_VOCAB,
        "direct_merge_pairs": len(merge_pairs),
        "splits": splits,
        "advance_to_probability_diagnostic": all(
            row["masked_true_pair_occurrences"] == 0 for row in splits.values()
        ),
        "no_test_access_or_scoring": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
