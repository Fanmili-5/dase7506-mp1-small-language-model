"""Verify a Git-less Stage143 scoring copy without opening any data split.

This is an asset-only staging check. It does not freeze the method, read test
text, or invoke the scorer. A later committed freeze record is still required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
FINAL = CODE / "results/stage143-evidence/final.json"
CHECKPOINT = "checkpoints/stage143-openvino-order6.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict:
    final = json.loads(FINAL.read_text(encoding="utf-8-sig"))
    if (final.get("protocol") != "7506-mp1-wt2-v2"
            or final.get("split") != "validation"
            or final.get("qualified") is not True
            or final.get("no_test_scoring") is not True):
        raise ValueError("Unexpected validation-only qualification")
    files = {CHECKPOINT: {
        "sha256": final["checkpoint_sha256"],
        "bytes": final["checkpoint_bytes"],
    }}
    for relative in final["inference_files"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in files:
            raise ValueError(f"Unsafe or duplicate inference file: {relative}")
        files[relative] = {
            "sha256": final["source_hashes"][relative],
            "bytes": final["asset_sizes"][relative],
        }
    if len(files) != 18 or sum(row["bytes"] for row in files.values()) != 55810412:
        raise ValueError("Qualified inference file set changed")
    if final["conservative_asset_bytes"] != 55810412:
        raise ValueError("Qualified asset total changed")
    checked = {}
    for relative, expected in sorted(files.items()):
        path = CODE / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing qualified file: {relative}")
        actual_bytes, actual_sha = path.stat().st_size, sha(path)
        if actual_bytes != expected["bytes"] or actual_sha != expected["sha256"]:
            raise ValueError(f"Qualified file changed: {relative}")
        checked[relative] = {"bytes": actual_bytes, "sha256": actual_sha}
    return {
        "status": "portable_assets_match_qualification_only",
        "protocol": final["protocol"],
        "qualification_sha256": sha(FINAL),
        "validation_bpb": final["validation_bpb"],
        "inference_files": len(checked),
        "conservative_inference_asset_bytes": sum(row["bytes"] for row in checked.values()),
        "checkpoint_sha256": checked[CHECKPOINT]["sha256"],
        "graph_sha256": checked["inference_assets/stage143-stage92-features.onnx"]["sha256"],
        "method_frozen": False,
        "test_scored_by_this_script": False,
        "test_text_opened_by_this_script": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new JSON output path")
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
