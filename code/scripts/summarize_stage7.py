"""Audit and summarize the Stage-7 dropout-boundary screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_stage3 import ROOT, audit
from summarize_stage6 import audit_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage7_summary.json")
    args = parser.parse_args()

    specs = [
        ("stage7-drop015-w256_d6-s17", "w256_d6_drop015", 0.15),
        ("stage7-drop020-w256_d6-s17", "w256_d6_drop020", 0.20),
    ]
    runs = []
    selected = []
    for name, method, dropout in specs:
        directory = ROOT / "runs" / name
        row = audit(directory, method, 17, 3600, 300)
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if row["parameters"] != 5_247_744:
            raise ValueError(f"Unexpected parameter count: {name}")
        if metrics["plan"]["config"].get("dropout") != dropout:
            raise ValueError(f"Unexpected dropout: {name}")
        runs.append(row)
        selected.append(audit_best(directory, row))

    stage6_path = ROOT.parents[2] / "outputs/windows-stage6-20260919/stage6_summary.json"
    stage6 = json.loads(stage6_path.read_text(encoding="utf-8"))
    control = stage6["winner"]
    candidates = [control, *({**row, "dropout": dropout} for row, dropout in zip(selected, (0.15, 0.20)))]
    winner = min(candidates, key=lambda row: row["validation_bpb"])
    if winner["dropout"] != 0.10:
        raise ValueError("Predeclared Stage-7 winner was not the dropout-0.10 control")

    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "dropout_010_control": control,
        "new_boundary_runs": runs,
        "new_validation_selected_runs": selected,
        "winner": winner,
        "dropout_015_regression_bpb": selected[0]["validation_bpb"] - control["validation_bpb"],
        "dropout_020_regression_bpb": selected[1]["validation_bpb"] - control["validation_bpb"],
        "resource_gate": stage6["resource_gate"],
        "new_training_targets": sum(row["train_targets"] for row in runs),
        "new_train_seconds": sum(row["train_seconds"] for row in runs),
        "new_process_seconds": sum(row["process_seconds"] for row in runs),
        "caveat": (
            "Validation only. The boundary screen rules out stronger dropout at this "
            "single-seed 3,600-step recipe; it does not replace cross-seed replication."
        ),
    }
    if result["new_training_targets"] != 58_982_400:
        raise ValueError("Stage-7 training budget mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
