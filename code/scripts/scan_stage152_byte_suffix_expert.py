"""Train-only raw-byte suffix to next-BPE-token diagnostic; no test access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_BYTES = (2, 3, 4)
PRIOR_STRENGTHS = (1.0, 10.0)
WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.20)
CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
EXPECTED_BPB = 1.399686162042141


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def byte_unicode_reverse() -> dict[str, int]:
    values = (list(range(ord("!"), ord("~") + 1))
              + list(range(ord("¡"), ord("¬") + 1))
              + list(range(ord("®"), ord("ÿ") + 1)))
    codepoints = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(values, codepoints)}


def encoded_stream(tokenizer: Tokenizer, path: Path,
                   token_bytes: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    ids = np.asarray(tokenizer.encode(raw.decode("utf-8")).ids, dtype=np.uint16)
    lengths = np.fromiter((len(token_bytes[int(token)]) for token in ids),
                          count=len(ids), dtype=np.int64)
    ends = lengths.cumsum()
    if (int(ends[-1]) != len(raw)
            or b"".join(token_bytes[int(token)] for token in ids) != raw):
        raise ValueError(f"BPE byte reconstruction mismatch: {path.name}")
    return ids, np.frombuffer(raw, dtype=np.uint8), ends


def suffix_keys(raw: np.ndarray, ends: np.ndarray,
                positions: np.ndarray, length: int) -> np.ndarray:
    key = np.zeros(len(positions), dtype=np.uint64)
    selected_ends = ends[positions]
    for lag in range(length, 0, -1):
        key = (key << np.uint64(8)) | raw[selected_ends - lag].astype(np.uint64)
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-logp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic output")
    cache = json.loads(args.target_logp.with_suffix(".json").read_text(encoding="utf-8"))
    if (cache.get("checkpoint_sha256") != CHECKPOINT_SHA
            or cache.get("array_sha256") != CACHE_SHA
            or sha(args.target_logp) != CACHE_SHA
            or cache.get("split") != "validation"
            or cache.get("test_scored") is not False):
        raise ValueError("Unexpected Stage143 target cache")
    logp = np.load(args.target_logp).astype(np.float64)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    paths = {split: ROOT / "data" / f"wikitext_{split}.txt"
             for split in ("train", "validation")}
    tokenizer_path = ROOT / "data/tokenizer.json"
    for path in (*paths.values(), tokenizer_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Fixed data/tokenizer changed: {path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    reverse = byte_unicode_reverse()
    token_bytes = [bytes(reverse[letter] for letter in tokenizer.id_to_token(token))
                   for token in range(2048)]
    train, train_raw, train_ends = encoded_stream(tokenizer, paths["train"], token_bytes)
    validation, val_raw, val_ends = encoded_stream(
        tokenizer, paths["validation"], token_bytes)
    if (len(logp) != len(validation) - 1 or not np.isfinite(logp).all()
            or train.max() >= 2048 or validation.max() >= 2048):
        raise ValueError("Token stream or cache shape changed")
    baseline = -float(logp.sum()) / np.log(2) / len(val_raw)
    if abs(baseline - EXPECTED_BPB) > 2e-5:
        raise ValueError("Stage143 BPB reproduction failed")
    unigram = np.bincount(train.astype(np.int64), minlength=2048).astype(np.float64) + 0.1
    unigram /= unigram.sum()
    val_inputs = np.arange(len(validation) - 1)
    val_starts = np.concatenate((np.zeros(1, dtype=np.int64), val_ends[:-1]))
    window_start_bytes = val_starts[(val_inputs // 256) * 256]
    result_rows = []
    for length in CONTEXT_BYTES:
        train_positions = np.flatnonzero(train_ends[:-1] >= length)
        valid = val_ends[:-1] - window_start_bytes >= length
        val_positions = val_inputs[valid]
        train_hist = suffix_keys(train_raw, train_ends, train_positions, length)
        val_hist = suffix_keys(val_raw, val_ends, val_positions, length)
        train_next = train[train_positions + 1].astype(np.uint64)
        val_truth = validation[val_positions + 1].astype(np.uint64)
        sorted_hist = np.sort(train_hist)
        denom = (np.searchsorted(sorted_hist, val_hist, side="right")
                 - np.searchsorted(sorted_hist, val_hist, side="left"))
        train_pairs = (train_hist << np.uint64(11)) | train_next
        val_pairs = (val_hist << np.uint64(11)) | val_truth
        sorted_pairs = np.sort(train_pairs)
        numer = (np.searchsorted(sorted_pairs, val_pairs, side="right")
                 - np.searchsorted(sorted_pairs, val_pairs, side="left"))
        if np.any(numer > denom):
            raise ValueError("Successor count exceeds context count")
        active = denom > 0
        active_positions = val_positions[active]
        reference = np.exp(logp[active_positions])
        for prior_strength in PRIOR_STRENGTHS:
            probability = (numer[active] + prior_strength * unigram[val_truth[active]]) \
                / (denom[active] + prior_strength)
            scores = []
            for weight in WEIGHTS:
                gain = (0.0 if weight == 0 else float(np.log(
                    (1 - weight) * reference + weight * probability
                ).sum() - logp[active_positions].sum()))
                scores.append({
                    "weight": weight,
                    "gain_nats": gain,
                    "validation_bpb": baseline - gain / np.log(2) / len(val_raw),
                })
            row = {
                "context_bytes": length,
                "prior_strength": prior_strength,
                "eligible_targets": int(len(val_positions)),
                "matched_targets": int(active.sum()),
                "correct_target_in_train_successors": int((numer > 0).sum()),
                "train_contexts": int(len(train_positions)),
                "scores": scores,
            }
            print(json.dumps(row), flush=True)
            result_rows.append(row)
    result = {
        "purpose": "target_only_train_byte_suffix_expert_diagnostic",
        "protocol": manifest["protocol"],
        "split": "validation",
        "test_scored": False,
        "source_sha256": sha(Path(__file__)),
        "target_cache_sha256": CACHE_SHA,
        "train_sha256": sha(paths["train"]),
        "validation_sha256": sha(paths["validation"]),
        "tokenizer_sha256": sha(tokenizer_path),
        "baseline_bpb": baseline,
        "rows": result_rows,
        "deployment_note": "Target-only diagnostic, not a full-vocabulary predictor",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
