"""Build canonical ASCII-letter BPE pair mask and audit train/validation incidence.

Never reads test, a model checkpoint, or validation labels during mask creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from audit_stage189_bpe_merge_pairs import (
    DATA, EXPECTED_TOKENIZER_SHA256, EXPECTED_VOCAB, PROTOCOL,
    audit_ids, sha,
)


def canonical_ascii_letter_mask(tokenizer: Tokenizer, vocab: dict[str, int]
                                ) -> tuple[set[tuple[int, int]], int]:
    if set(vocab.values()) != set(range(EXPECTED_VOCAB)):
        raise ValueError("Unexpected vocabulary IDs")
    symbols = sorted((token_id, token) for token, token_id in vocab.items()
                     if token and token.isascii() and token.isalpha())
    forbidden = set()
    for left_id, left in symbols:
        for right_id, right in symbols:
            if tokenizer.encode(left + right).ids != [left_id, right_id]:
                forbidden.add((left_id, right_id))
    return forbidden, len(symbols)


def pair_sha(pairs: set[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for left, right in sorted(pairs):
        digest.update(left.to_bytes(2, "little"))
        digest.update(right.to_bytes(2, "little"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    token_path = DATA / "tokenizer.json"
    if (manifest["protocol"] != PROTOCOL
            or sha(token_path) != EXPECTED_TOKENIZER_SHA256
            or manifest["sha256"]["tokenizer.json"] != EXPECTED_TOKENIZER_SHA256):
        raise ValueError("Unexpected fixed tokenizer/protocol")
    tokenizer_json = json.loads(token_path.read_text(encoding="utf-8"))
    if (tokenizer_json["model"]["type"] != "BPE"
            or tokenizer_json["pre_tokenizer"]["type"] != "ByteLevel"):
        raise ValueError("Unexpected tokenizer structure")
    tokenizer = Tokenizer.from_file(str(token_path))
    mask, symbol_count = canonical_ascii_letter_mask(
        tokenizer, tokenizer_json["model"]["vocab"])
    print(f"{symbol_count} ASCII letter symbols; {len(mask)} noncanonical pairs", flush=True)
    splits = {}
    for split in ("train", "validation"):
        name = f"wikitext_{split}.txt"
        path = DATA / name
        if sha(path) != manifest["sha256"][name]:
            raise ValueError(f"Fixed {split} text hash changed")
        raw = path.read_bytes()
        ids = tokenizer.encode(raw.decode("utf-8")).ids
        splits[split] = {"utf8_bytes": len(raw), "text_sha256": sha(path),
                         **audit_ids(ids, mask)}
        print(f"{split}: {splits[split]['masked_true_pair_occurrences']} masked true pairs", flush=True)
    result = {
        "protocol": PROTOCOL,
        "purpose": "stage190_ascii_letter_pair_incidence_only",
        "source_sha256": sha(Path(__file__)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "ascii_letter_symbols": symbol_count,
        "ordered_ascii_letter_pairs": symbol_count ** 2,
        "noncanonical_mask_pairs": len(mask),
        "mask_pair_sha256": pair_sha(mask),
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
