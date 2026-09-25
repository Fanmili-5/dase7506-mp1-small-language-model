"""Train-only skip-context target-probability diagnostic; never score test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ((0, 2), (0, 2, 4), (0, 1, 3), (0, 1, 4), (0, 1, 2, 4))
WEIGHTS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
EXPECTED_CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
EXPECTED_BASE_BPB = 1.399686162042141
EXPECTED_CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_history(tokens: np.ndarray, ends: np.ndarray,
                    pattern: tuple[int, ...]) -> np.ndarray:
    """Losslessly pack up to four 11-bit BPE IDs into a uint64 key."""
    key = np.zeros(len(ends), dtype=np.uint64)
    for lag in pattern:
        key = (key << np.uint64(11)) | tokens[ends - lag].astype(np.uint64)
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-logp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic output")
    cache = json.loads(args.target_logp.with_suffix(".json").read_text(encoding="utf-8"))
    if (cache.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA
            or cache.get("array_sha256") != EXPECTED_CACHE_SHA
            or sha(args.target_logp) != EXPECTED_CACHE_SHA
            or cache.get("split") != "validation"
            or cache.get("test_scored") is not False):
        raise ValueError("Unexpected target-probability cache")
    logp = np.load(args.target_logp).astype(np.float64)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    paths = {split: ROOT / "data" / f"wikitext_{split}.txt"
             for split in ("train", "validation")}
    tokenizer_path = ROOT / "data/tokenizer.json"
    for path in (*paths.values(), tokenizer_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Fixed data/tokenizer changed: {path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train = np.asarray(tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids,
                       dtype=np.uint16)
    validation = np.asarray(
        tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids,
        dtype=np.uint16)
    if (len(logp) != len(validation) - 1 or not np.isfinite(logp).all()
            or train.max() >= 2048 or validation.max() >= 2048):
        raise ValueError("Token stream or cache shape changed")
    byte_count = paths["validation"].stat().st_size
    baseline_bpb = -float(logp.sum()) / np.log(2) / byte_count
    if abs(baseline_bpb - EXPECTED_BASE_BPB) > 2e-5:
        raise ValueError("Stage143 validation cache failed reproduction")

    rows = []
    for pattern in PATTERNS:
        maximum_lag = max(pattern)
        train_ends = np.arange(maximum_lag, len(train) - 1)
        validation_ends = np.arange(maximum_lag, len(validation) - 1)
        validation_ends = validation_ends[
            validation_ends % 256 >= maximum_lag]
        train_hist = encoded_history(train, train_ends, pattern)
        val_hist = encoded_history(validation, validation_ends, pattern)
        truth = validation[validation_ends + 1].astype(np.uint64)
        sorted_hist = np.sort(train_hist)
        denom = (np.searchsorted(sorted_hist, val_hist, side="right")
                 - np.searchsorted(sorted_hist, val_hist, side="left"))
        train_pair = (train_hist << np.uint64(11)) | train[train_ends + 1].astype(np.uint64)
        val_pair = (val_hist << np.uint64(11)) | truth
        sorted_pair = np.sort(train_pair)
        numer = (np.searchsorted(sorted_pair, val_pair, side="right")
                 - np.searchsorted(sorted_pair, val_pair, side="left"))
        if np.any(numer > denom):
            raise ValueError("Successor count exceeds context count")
        active = denom > 0
        target_logp = logp[validation_ends[active]]
        successor_probability = numer[active] / denom[active]
        scores = []
        for weight in WEIGHTS:
            gain_nats = (0.0 if weight == 0 else float(np.log(
                (1 - weight) * np.exp(target_logp)
                + weight * successor_probability).sum() - target_logp.sum()))
            scores.append({
                "weight": weight,
                "gain_nats": gain_nats,
                "validation_bpb": baseline_bpb - gain_nats / np.log(2) / byte_count,
            })
        row = {
            "pattern_lags": list(pattern),
            "eligible_targets": int(len(validation_ends)),
            "matched_targets": int(active.sum()),
            "correct_target_in_train_successors": int((numer > 0).sum()),
            "scores": scores,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "purpose": "target_only_train_skip_context_mixture_diagnostic",
        "protocol": manifest["protocol"],
        "split": "validation",
        "test_scored": False,
        "source_sha256": sha(Path(__file__)),
        "target_cache_sha256": sha(args.target_logp),
        "train_sha256": sha(paths["train"]),
        "validation_sha256": sha(paths["validation"]),
        "tokenizer_sha256": sha(tokenizer_path),
        "baseline_bpb": baseline_bpb,
        "rows": rows,
        "deployment_note": "Target-only scan is not a normalized predictor or qualified checkpoint",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
