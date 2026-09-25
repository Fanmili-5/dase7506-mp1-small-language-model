"""Read-only Stage143 freeze preflight; write a freeze record only with --confirm.

This script never opens or scores the test split. Running it without
--confirm only reports whether the currently committed candidate is ready
to freeze; it does not freeze the method or authorize test evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from audit_stage143_pretest_readiness import audit as pretest_audit

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
FINAL = CODE / "results/stage143-evidence/final.json"
CHECKPOINT = "checkpoints/stage143-openvino-order6.pt"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> dict:
    readiness = pretest_audit()
    final = json.loads(FINAL.read_text(encoding="utf-8-sig"))
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    files = {CHECKPOINT: final["checkpoint_sha256"]}
    files.update(final["source_hashes"])
    tracked = {}
    for relative, expected in sorted(files.items()):
        path = CODE / relative
        if sha(path) != expected:
            raise ValueError(f"Candidate file differs from qualification: {relative}")
        repo_path = "code/" + relative
        try:
            blob_bytes = int(git("cat-file", "-s", f"HEAD:{repo_path}"))
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Candidate file is not committed: {repo_path}") from error
        if blob_bytes != path.stat().st_size:
            raise ValueError(f"Committed blob size differs: {repo_path}")
        tracked[relative] = {"sha256": expected, "bytes": path.stat().st_size}
    if sum(row["bytes"] for row in tracked.values()) != final["conservative_asset_bytes"]:
        raise ValueError("Inference asset sum changed")
    return {
        "status": "eligible_for_freeze_only" if not dirty else "working_tree_dirty_not_ready",
        "method_frozen": False,
        "test_scored_by_this_script": False,
        "protocol": final["protocol"],
        "candidate": "stage143-openvino-order6",
        "source_commit": commit,
        "qualification_sha256": sha(FINAL),
        "validation_bpb": readiness["windows_validation_bpb"],
        "windows_cpu_time_ratio": readiness["windows_cpu_time_ratio"],
        "windows_peak_rss_bytes": readiness["windows_peak_rss_bytes"],
        "conservative_inference_asset_bytes": readiness["conservative_inference_asset_bytes"],
        "inference_files": tracked,
        "dirty_entries": dirty.splitlines(),
        "next_action": "Explicitly decide to freeze before any Stage143 test evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="Record an explicit method freeze before test scoring")
    parser.add_argument("--output", type=Path,
                        help="New JSON path for the freeze record; required with --confirm")
    args = parser.parse_args()
    if args.output and not args.confirm:
        parser.error("--output is only for an explicit --confirm freeze")
    if args.confirm and not args.output:
        parser.error("--confirm requires a new --output path")
    result = preflight()
    if args.confirm:
        if result["status"] != "eligible_for_freeze_only":
            raise ValueError("Cannot freeze a dirty working tree")
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite freeze record")
        result["status"] = "method_frozen_before_test"
        result["method_frozen"] = True
        result["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["next_action"] = "Score the exact frozen predictor on complete CPU FP32 test once."
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
