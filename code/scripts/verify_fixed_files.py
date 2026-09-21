"""Verify the course-controlled evaluator, data, tokenizer, and baseline files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
FIXED = [
    "common.py",
    "evaluate.py",
    "model.py",
    "configs/baseline.json",
    "data/manifest.json",
    "data/tokenizer.json",
    "data/wikitext_train.txt",
    "data/wikitext_validation.txt",
    "data/wikitext_test.txt",
    "tests/test_contract.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    failures = []
    for relative in FIXED:
        path = ROOT / relative
        expected = MANIFEST[f"code/{relative}"]
        actual = digest(path)
        status = "OK" if actual == expected else "CHANGED"
        print(f"{status:7} {relative}")
        if actual != expected:
            failures.append(relative)
    if failures:
        raise SystemExit("Fixed-file verification failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
