"""Locate Stage143 mid-frequency/unseen-prefix validation loss, without test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            or cache.get("split") != "validation"):
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
    train = np.asarray(tokenizer.encode(paths["train"].read_text(encoding="utf-8")).ids)
    validation = np.asarray(
        tokenizer.encode(paths["validation"].read_text(encoding="utf-8")).ids)
    if len(logp) != len(validation) - 1 or not np.isfinite(logp).all():
        raise ValueError("Validation cache length changed")
    train_frequency = np.bincount(train, minlength=2048)
    targets = validation[1:]
    mid_frequency = (train_frequency[targets] >= 100) & (train_frequency[targets] < 1000)
    seen = np.zeros(len(targets), dtype=np.bool_)
    prefix: set[int] = set()
    for input_index, target in enumerate(targets):
        if input_index % 256 == 0:
            prefix.clear()
        prefix.add(int(validation[input_index]))
        seen[input_index] = int(target) in prefix
    nll = -logp
    groups = {}
    for mid_label, mid_mask in (("mid_frequency_100_999", mid_frequency),
                                ("all_other_frequency", ~mid_frequency)):
        for seen_label, seen_mask in (("seen_in_prefix", seen),
                                      ("unseen_in_prefix", ~seen)):
            mask = mid_mask & seen_mask
            count = int(mask.sum())
            loss = float(nll[mask].sum())
            groups[f"{mid_label}/{seen_label}"] = {
                "targets": count,
                "nll_nats": loss,
                "mean_nll_nats": loss / count if count else None,
            }
    bytes_count = paths["validation"].stat().st_size
    target_gain_bpb = 0.05
    required_nats = target_gain_bpb * bytes_count * math.log(2)
    mid_unseen = groups["mid_frequency_100_999/unseen_in_prefix"]
    result = {
        "purpose": "validation_error_allocation_not_model_selection",
        "protocol": manifest["protocol"],
        "split": "validation",
        "test_scored": False,
        "source_sha256": sha(Path(__file__)),
        "target_cache_sha256": CACHE_SHA,
        "train_sha256": sha(paths["train"]),
        "validation_sha256": sha(paths["validation"]),
        "tokenizer_sha256": sha(tokenizer_path),
        "validation_targets": len(targets),
        "validation_bpb": float(nll.sum() / math.log(2) / bytes_count),
        "groups": groups,
        "gap_probe": {
            "desired_bpb_reduction": target_gain_bpb,
            "total_nats_reduction_required": required_nats,
            "mean_nats_reduction_required_if_only_mid_frequency_unseen_improves":
                required_nats / mid_unseen["targets"],
            "warning": "Accounting identity, not a feasible prediction or oracle score",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
