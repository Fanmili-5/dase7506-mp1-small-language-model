"""Retrospective train/validation-only lexical allocation of Stage143/155 gain."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
OLD_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
NEW_SHA = "0cbe8ee4aec517ba756fc5e167a95aea2b82e088b5f870e29f77337a9a499603"
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007
START = re.compile(r"Ġ[A-Za-z]+\Z")
CONTINUATION = re.compile(r"[A-Za-z]+\Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spelling_type(spelling: str) -> str:
    if START.fullmatch(spelling):
        return "ascii_word_start"
    if CONTINUATION.fullmatch(spelling):
        return "ascii_word_continuation"
    return "other"


def grouped_stats(mask: np.ndarray, old: np.ndarray,
                  mixture: np.ndarray, scale: float) -> dict:
    count = int(mask.sum())
    old_nll = -float(old[mask].sum())
    gain = float((mixture[mask] - old[mask]).sum())
    return {
        "targets": count,
        "stage143_nll_nats": old_nll,
        "stage143_mean_nll_nats": old_nll / count if count else None,
        "mixture_gain_nats": gain,
        "mixture_gain_bpb": gain / scale,
    }


def analyze(old: np.ndarray, new: np.ndarray, train: np.ndarray,
            validation: np.ndarray, spellings: list[str], byte_count: int) -> dict:
    if (old.shape != new.shape or old.shape != (EXPECTED_TARGETS,)
            or validation.shape != (EXPECTED_TARGETS + 1,)
            or byte_count != EXPECTED_BYTES or len(spellings) != 2048
            or not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError("Incomplete or invalid fixed validation inputs")
    targets = validation[1:]
    frequency = np.bincount(train, minlength=2048)
    mid = (frequency[targets] >= 100) & (frequency[targets] < 1000)
    seen = np.zeros(EXPECTED_TARGETS, dtype=np.bool_)
    prefix: set[int] = set()
    for index, target in enumerate(targets):
        if index % 256 == 0:
            prefix.clear()
        prefix.add(int(validation[index]))
        seen[index] = int(target) in prefix
    spelling_codes = np.asarray([{"ascii_word_start": 0,
                                  "ascii_word_continuation": 1,
                                  "other": 2}[spelling_type(value)]
                                 for value in spellings], dtype=np.int8)
    types = spelling_codes[targets]
    mixture = np.logaddexp(old, new) - math.log(2)
    scale = math.log(2) * byte_count
    groups = {}
    marginals = {}
    names = ("ascii_word_start", "ascii_word_continuation", "other")
    for code, name in enumerate(names):
        lexical = types == code
        marginals[name] = grouped_stats(lexical, old, mixture, scale)
        for freq_name, freq_mask in (("mid_frequency_100_999", mid),
                                     ("other_frequency", ~mid)):
            for seen_name, seen_mask in (("seen_in_prefix", seen),
                                          ("unseen_in_prefix", ~seen)):
                key = f"{name}/{freq_name}/{seen_name}"
                groups[key] = grouped_stats(
                    lexical & freq_mask & seen_mask, old, mixture, scale)
    if sum(value["targets"] for value in groups.values()) != EXPECTED_TARGETS:
        raise AssertionError("Groups do not partition validation targets")
    gain = float((mixture - old).sum() / scale)
    if abs(sum(value["mixture_gain_bpb"] for value in groups.values()) - gain) > 1e-10:
        raise AssertionError("Group gains do not reconstruct total mixture gain")
    old_bpb = -float(old.sum()) / scale
    if abs(old_bpb - 1.399686162042141) > 2e-5 or abs(gain - 0.0235034978) > 2e-5:
        raise ValueError("Paired caches do not reproduce archived Stage162 scores")
    return {
        "purpose": "retrospective_validation_error_allocation_not_inference_gate",
        "split": "validation", "no_test_scoring": True,
        "targets": EXPECTED_TARGETS, "utf8_bytes": byte_count,
        "stage143_bpb": old_bpb, "equal_mixture_bpb": old_bpb - gain,
        "equal_mixture_gain_bpb": gain,
        "marginals": marginals, "groups": groups,
        "warning": "True-target groups and Stage155 teacher gains cannot be used at inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new diagnostic output")
    old_path = ROOT / "results/stage162-evidence/stage143-target-logp.npy"
    new_path = ROOT / "results/stage162-evidence/stage155-target-logp.npy"
    if sha(old_path) != OLD_SHA or sha(new_path) != NEW_SHA:
        raise ValueError("Paired frozen validation caches changed")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    data_paths = [ROOT / "data" / name for name in
                  ("tokenizer.json", "wikitext_train.txt", "wikitext_validation.txt")]
    for path in data_paths:
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Fixed course input changed: {path.name}")
    tokenizer = Tokenizer.from_file(str(data_paths[0]))
    train = np.asarray(tokenizer.encode(data_paths[1].read_text(encoding="utf-8")).ids)
    validation = np.asarray(tokenizer.encode(data_paths[2].read_text(encoding="utf-8")).ids)
    vocabulary = json.loads(data_paths[0].read_text(encoding="utf-8"))["model"]["vocab"]
    spellings = [""] * 2048
    for spelling, token_id in vocabulary.items():
        spellings[token_id] = spelling
    result = analyze(np.load(old_path).astype(np.float64),
                     np.load(new_path).astype(np.float64), train, validation,
                     spellings, data_paths[2].stat().st_size)
    result.update(source_sha256=sha(Path(__file__)), old_cache_sha256=OLD_SHA,
                  new_cache_sha256=NEW_SHA,
                  tokenizer_sha256=sha(data_paths[0]),
                  train_sha256=sha(data_paths[1]),
                  validation_sha256=sha(data_paths[2]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
