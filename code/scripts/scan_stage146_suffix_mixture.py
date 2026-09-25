"""Score train-derived long-suffix target probabilities on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ORDERS = (6, 7, 8)
WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.20)
BASE = np.uint64(11400714819323198485)
PAIR_MASK = np.uint64((1 << 53) - 1)
EXPECTED_BASE_BPB = 1.399686162042141


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-logp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new diagnostic output")
    cache = json.loads(args.target_logp.with_suffix(".json").read_text(encoding="utf-8"))
    if (cache["checkpoint_sha256"] !=
            "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
            or cache["array_sha256"] != sha(args.target_logp)
            or cache["split"] != "validation"):
        raise ValueError("Target-probability cache changed")
    logp = np.load(args.target_logp).astype(np.float64)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    paths = {name: ROOT / "data" / f"wikitext_{name}.txt"
             for name in ("train", "validation")}
    tokenizer_path = ROOT / "data/tokenizer.json"
    for path in (*paths.values(), tokenizer_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Fixed data/tokenizer changed: {path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train = np.asarray(tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids,
                       dtype=np.uint64)
    val = np.asarray(tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids,
                     dtype=np.uint64)
    if len(logp) != len(val) - 1:
        raise ValueError("Array and validation target count differ")
    byte_count = paths["validation"].stat().st_size
    baseline_bpb = -float(logp.sum()) / np.log(2) / byte_count
    if abs(baseline_bpb - EXPECTED_BASE_BPB) > 2e-5:
        raise ValueError("Baseline target probability cache does not reproduce Stage143")
    th, vh = train + 1, val + 1
    rows = []
    for order in range(1, max(ORDERS) + 1):
        if order > 1:
            th = th[:-1] * BASE + train[order - 1:] + 1
            vh = vh[:-1] * BASE + val[order - 1:] + 1
        if order not in ORDERS:
            continue
        train_hist, train_next = th[:-1], train[order:]
        ends = np.arange(order - 1, len(val) - 1)
        allowed = ends % 256 >= order - 1
        positions = ends[allowed]
        query = vh[:-1][allowed]
        truth = val[order:][allowed]
        sorted_hist = np.sort(train_hist)
        denom = (np.searchsorted(sorted_hist, query, side="right")
                 - np.searchsorted(sorted_hist, query, side="left"))
        train_pairs = ((train_hist & PAIR_MASK) << np.uint64(11)) | train_next
        query_pairs = ((query & PAIR_MASK) << np.uint64(11)) | truth
        sorted_pairs = np.sort(train_pairs)
        numer = (np.searchsorted(sorted_pairs, query_pairs, side="right")
                 - np.searchsorted(sorted_pairs, query_pairs, side="left"))
        if np.any(numer > denom):
            raise ValueError("Hash collision or count alignment error")
        active = denom > 0
        nll = logp[positions[active]]
        retrieval = numer[active] / denom[active]
        scores = []
        for weight in WEIGHTS:
            if weight == 0:
                gain = 0.0
            else:
                changed = np.log((1 - weight) * np.exp(nll) + weight * retrieval)
                gain = float(np.sum(changed - nll))
            scores.append({
                "weight": weight,
                "gain_nats": gain,
                "validation_bpb": baseline_bpb - gain / np.log(2) / byte_count,
            })
        row = {
            "order": order,
            "eligible_targets": int(len(positions)),
            "active_targets": int(np.count_nonzero(active)),
            "correct_target_in_train_successors": int(np.count_nonzero(numer > 0)),
            "scores": scores,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "purpose": "target_only_train_suffix_mixture_diagnostic",
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
        "deployment_note": "This target-only scan is not a predictor or resource-qualified artifact",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
