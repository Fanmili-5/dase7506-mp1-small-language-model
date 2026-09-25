"""Run the fixed CPU FP32 test scorer only against a committed Stage143 freeze.

Without a committed freeze record this script refuses to invoke evaluate.py.
It does not import, inspect, or cache the test split during preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_stage143_final_release import (
    CHECKPOINT, CODE, FINAL, REPO, expected_files, git, read_json, sha,
    validate_freeze, validate_test,
)


def preflight(freeze_path: Path, output: Path) -> tuple[dict, dict, str]:
    if not freeze_path.is_file():
        raise ValueError("No method freeze record exists; refusing to score test")
    if output.suffix.lower() != ".json" or output.exists():
        raise ValueError("Choose a new JSON result path; existing results are never overwritten")
    if output.with_suffix(".window-nll.npy").exists():
        raise ValueError("A window-loss artifact already exists at the chosen result path")
    try:
        freeze_relative = freeze_path.resolve().relative_to(REPO.resolve()).as_posix()
        output.resolve().relative_to(REPO.resolve())
    except ValueError as error:
        raise ValueError("Freeze and result paths must be inside the repository") from error
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Commit the freeze record and leave the worktree clean before test")
    freeze_bytes = freeze_path.read_bytes()
    try:
        committed_bytes = subprocess.check_output(
            ["git", "show", f"HEAD:{freeze_relative}"], cwd=REPO,
            stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError("Commit the freeze record before scoring test") from error
    if freeze_bytes != committed_bytes:
        raise ValueError("The freeze record is not committed unchanged at HEAD")
    final, freeze = read_json(FINAL), read_json(freeze_path)
    files = validate_freeze(freeze, final, sha(FINAL))
    subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"],
                    git("rev-parse", "HEAD")], cwd=REPO, check=True)
    for relative, row in files.items():
        path = CODE / relative
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen inference file changed: {relative}")
    if files != expected_files(final):
        raise ValueError("Frozen file set differs from qualification")
    return final, freeze, sha(freeze_path)


def run(freeze_path: Path, output: Path) -> dict:
    final, freeze, freeze_sha = preflight(freeze_path, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(CODE / "evaluate.py"),
        "--checkpoint", str(CODE / CHECKPOINT),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "test", "--output", str(output),
    ], cwd=CODE, check=True)
    result = read_json(output)
    result["freeze_record_sha256"] = freeze_sha
    result["frozen_source_commit"] = freeze["source_commit"]
    result["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    validate_test(result, final, freeze_sha, freeze["source_commit"])
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.output)
    print(json.dumps({
        "status": "complete_test_scored_after_committed_freeze",
        "bpb": result["bpb"],
        "targets": result["targets"],
        "utf8_bytes": result["utf8_bytes"],
        "freeze_record_sha256": result["freeze_record_sha256"],
        "result_sha256": sha(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
