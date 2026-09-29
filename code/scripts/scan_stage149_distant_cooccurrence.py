"""Train-only distant-cooccurrence target diagnostic; no test access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
VOCAB = 2048
RANGES = ((5, 16), (5, 32))
PRIOR_STRENGTHS = (100, 1000)
WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.20)
EXPECTED_CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
EXPECTED_CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
EXPECTED_BPB = 1.399686162042141


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_pair_counts(counts: np.ndarray, tokens: np.ndarray, lag: int) -> None:
    """Count train pairs: token at i-lag, successor token at i+1."""
    left = tokens[:-lag - 1].astype(np.int64)
    right = tokens[lag + 1:].astype(np.int64)
    pairs = left * VOCAB + right
    counts += np.bincount(pairs, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB).astype(np.int32)


def target_probability(counts: np.ndarray, unigram: np.ndarray,
                       validation: np.ndarray, max_lag: int,
                       prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
    ends = np.arange(len(validation) - 1)
    next_tokens = validation[ends + 1].astype(np.int64)
    rows = counts.sum(axis=1, dtype=np.int64)
    probability = np.zeros(len(ends), dtype=np.float64)
    eligible = np.zeros(len(ends), dtype=np.int16)
    for lag in range(5, max_lag + 1):
        active = ends % 256 >= lag
        positions = ends[active]
        x = validation[positions - lag].astype(np.int64)
        y = next_tokens[active]
        probability[active] += (
            counts[x, y] + prior_strength * unigram[y]
        ) / (rows[x] + prior_strength)
        eligible[active] += 1
    active = eligible > 0
    probability[active] /= eligible[active]
    if (np.any(probability[active] <= 0)
            or np.any(probability[active] > 1)):
        raise ValueError("Invalid co-occurrence probability")
    return probability, active


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
    train = np.asarray(tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids,
                       dtype=np.uint16)
    validation = np.asarray(
        tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids,
        dtype=np.uint16)
    if (len(logp) != len(validation) - 1 or not np.isfinite(logp).all()
            or train.max() >= VOCAB or validation.max() >= VOCAB):
        raise ValueError("Token stream or target cache changed")
    byte_count = paths["validation"].stat().st_size
    baseline = -float(logp.sum()) / np.log(2) / byte_count
    if abs(baseline - EXPECTED_BPB) > 2e-5:
        raise ValueError("Stage143 BPB reproduction failed")
    unigram = (np.bincount(train.astype(np.int64), minlength=VOCAB) + 0.1)
    unigram = unigram / unigram.sum()
    counts = np.zeros((VOCAB, VOCAB), dtype=np.int32)
    result_rows = []
    for lag in range(5, RANGES[-1][1] + 1):
        add_pair_counts(counts, train, lag)
        if lag not in (16, 32):
            continue
        for prior_strength in PRIOR_STRENGTHS:
            expert, active = target_probability(
                counts, unigram, validation, lag, prior_strength)
            reference = np.exp(logp[active])
            scores = []
            for weight in WEIGHTS:
                gain = (0.0 if weight == 0 else float(np.log(
                    (1 - weight) * reference + weight * expert[active]
                ).sum() - logp[active].sum()))
                scores.append({
                    "weight": weight,
                    "gain_nats": gain,
                    "validation_bpb": baseline - gain / np.log(2) / byte_count,
                })
            row = {
                "distance_range": [5, lag],
                "prior_strength": prior_strength,
                "active_targets": int(active.sum()),
                "train_pair_occurrences": int(counts.sum(dtype=np.int64)),
                "scores": scores,
            }
            print(json.dumps(row), flush=True)
            result_rows.append(row)

    result = {
        "purpose": "target_only_train_distant_cooccurrence_diagnostic",
        "protocol": manifest["protocol"],
        "split": "validation",
        "test_scored": False,
        "source_sha256": sha(Path(__file__)),
        "target_cache_sha256": sha(args.target_logp),
        "train_sha256": sha(paths["train"]),
        "validation_sha256": sha(paths["validation"]),
        "tokenizer_sha256": sha(tokenizer_path),
        "baseline_bpb": baseline,
        "rows": result_rows,
        "deployment_note": "Target-only diagnostic; no full-vocabulary predictor or resource qualification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
