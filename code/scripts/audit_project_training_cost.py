"""Summarize archived training-time records without claiming a complete bill."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    rows = []
    seen_checkpoints = {}
    for metrics_path in sorted(RESULTS.glob("**/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        seconds = metrics.get("train_seconds")
        if seconds is not None and (
            not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(f"Invalid train_seconds: {metrics_path}")
        relative = metrics_path.relative_to(ROOT).as_posix()
        checkpoint = metrics.get("checkpoint_sha256")
        if checkpoint:
            previous = seen_checkpoints.get(checkpoint)
            if previous is not None:
                raise ValueError(
                    f"Checkpoint is recorded twice: {previous} and {relative}")
            seen_checkpoints[checkpoint] = relative
        run_path = metrics_path.with_name("run.json")
        rows.append({
            "metrics_path": relative,
            "metrics_sha256": digest(metrics_path),
            "run_path": run_path.relative_to(ROOT).as_posix() if run_path.exists() else None,
            "run_sha256": digest(run_path) if run_path.exists() else None,
            "reported_device": metrics.get("device"),
            "train_seconds": seconds,
            "other_reported_elapsed_seconds": metrics.get("elapsed_seconds"),
            "checkpoint_sha256": checkpoint,
        })
    recorded_seconds = [row["train_seconds"] for row in rows
                        if row["train_seconds"] is not None]
    return {
        "purpose": "archived_training_time_record_sum_not_complete_project_search_cost",
        "metrics_records": len(rows),
        "positive_train_seconds_records": sum(seconds > 0 for seconds in recorded_seconds),
        "missing_train_seconds_record_paths": [row["metrics_path"] for row in rows
                                               if row["train_seconds"] is None],
        "raw_sum_reported_train_seconds": sum(recorded_seconds),
        "raw_sum_reported_train_hours": sum(recorded_seconds) / 3600,
        "records": rows,
        "caveats": [
            "This is a deterministic sum of archived metrics.json train_seconds fields; it is not a complete project-wide search-cost total.",
            "Checkpoint SHA-256 duplication is rejected when hashes are present; records without checkpoint hashes are identified only by their distinct evidence paths.",
            "Records without a train_seconds field are listed but excluded from the sum even if they report a broader elapsed_seconds field.",
            "Missing, failed-before-metrics, external, and manual experiments; validation, data construction, packaging, CPU checks and setup are excluded.",
            "Reported train_seconds is taken as recorded by each original script and is not normalized for GPU utilization or stochastic forward count.",
        ],
    }


def main() -> None:
    result = audit()
    output = RESULTS / "project-training-cost-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
