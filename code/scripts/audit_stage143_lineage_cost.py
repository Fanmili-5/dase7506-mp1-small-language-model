"""Audit recorded accepted Stage143 neural-lineage training cost, without guessing search cost."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = (
    (54, "primary_targets", 58_982_400),
    (56, "additional_primary_targets", 39_321_600),
    (61, "additional_primary_targets", 39_321_600),
    (63, "additional_primary_targets", 29_491_200),
    (65, "additional_primary_targets", 29_491_200),
    (67, "total_train_target_presentations", 18_066_710),
    (71, "primary_targets", 29_491_200),
    (92, "primary_targets", 11_059_200),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for stage, target_key, expected in RECORDS:
        folder = ROOT / f"results/stage{stage}-evidence"
        run_path, metrics_path = folder / "run.json", folder / "metrics.json"
        run = json.loads(run_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        target_count = run.get(target_key)
        train_seconds = metrics.get("train_seconds")
        validation_seconds = metrics.get("validation_seconds")
        if (target_count != expected or not isinstance(train_seconds, (int, float))
                or train_seconds <= 0 or not isinstance(validation_seconds, (int, float))
                or validation_seconds <= 0):
            raise ValueError(f"Invalid accepted-lineage record at Stage{stage}")
        rows.append(dict(stage=stage, target_field=target_key,
                         primary_training_target_presentations=target_count,
                         train_seconds=train_seconds,
                         validation_seconds=validation_seconds,
                         run_sha256=digest(run_path),
                         metrics_sha256=digest(metrics_path)))
    result = dict(
        purpose="accepted_neural_lineage_cost_lower_bound_not_total_search",
        stages=[row["stage"] for row in rows], records=rows,
        cumulative_primary_training_target_presentations=sum(
            row["primary_training_target_presentations"] for row in rows),
        cumulative_recorded_train_seconds=sum(row["train_seconds"] for row in rows),
        cumulative_recorded_validation_seconds=sum(row["validation_seconds"] for row in rows),
        caveats=[
            "Stage67 counts each full training pass's labels; other stages report their documented primary targets.",
            "R-Drop causes two stochastic presentations of many primary targets and raises training compute; the primary count is not a full FLOP measure.",
            "This excludes count-table construction, rejected/failed search runs, preflights, export, CPU audits and general overhead.",
            "Stages54/56/61/63/65/67/71/92 are the accepted neural-weight lineage feeding Stage143; do not interpret these sums as total project search cost.",
        ],
    )
    output = ROOT / "results/stage143-evidence/lineage-cost.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
