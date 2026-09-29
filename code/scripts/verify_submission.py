"""Read-only checks of frozen assets, recorded score and release identity; no scoring."""
from __future__ import annotations

import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts.package_stage143_final_release import (
    FINAL, qualification_hashes, read_json, sha, validate_freeze, validate_test)


def verify():
    final = read_json(FINAL)
    freeze_path = FINAL.parent / "freeze-stage143-20260927.json"
    freeze = read_json(freeze_path)
    test = read_json(FINAL.parent / "test-stage143-20260927.json")
    files = validate_freeze(freeze, final, qualification_hashes(FINAL))
    validate_test(test, final, sha(freeze_path), freeze["source_commit"])
    for relative, expected in files.items():
        path = CODE / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {relative}; extract the matching release ZIP first")
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Frozen file changed: {relative}")
    manifest_path = CODE.parent / "BUNDLE_MANIFEST.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["files"] != files:
            raise ValueError("Bundle file manifest differs from the frozen inference set")
        if sha(CODE.parent / "REPORT.pdf") != manifest["report_sha256"]:
            raise ValueError("Report differs from the bundle release")
    return {"frozen_files_verified": len(files),
            "inference_asset_bytes": sum(row["bytes"] for row in files.values()),
            "recorded_test_bpb": test["bpb"], "scored_by_this_check": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
