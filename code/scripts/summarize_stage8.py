"""Audit and summarize the Stage-8 duration screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_stage3 import ROOT, audit
from summarize_stage6 import audit_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage8_summary.json")
    args = parser.parse_args()

    directory = ROOT / "runs/stage8-long-drop010-w256_d6-s17"
    endpoint = audit(directory, "w256_d6_drop010", 17, 4800, 300)
    if endpoint["parameters"] != 5_247_744:
        raise ValueError("Unexpected parameter count")
    selected = audit_best(directory, endpoint)
    if selected["step"] != 4800:
        raise ValueError("Stage-8 selected checkpoint was not the endpoint")

    stage6_path = ROOT.parents[2] / "outputs/windows-stage6-20260919/stage6_summary.json"
    stage6 = json.loads(stage6_path.read_text(encoding="utf-8"))
    control = stage6["winner"]
    gain = control["validation_bpb"] - selected["validation_bpb"]
    if gain < 0.002:
        raise ValueError("Long recipe did not pass its preregistered adoption threshold")

    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "long_run": endpoint,
        "validation_selected": selected,
        "stage6_3600_step_control": control,
        "gain_vs_3600_step_control_bpb": gain,
        "decision": "Adopt the 4,800-step dropout-0.10 recipe for seeds 23 and 42.",
        "resource_gate": stage6["resource_gate"],
        "new_training_targets": endpoint["train_targets"],
        "new_train_seconds": endpoint["train_seconds"],
        "new_process_seconds": endpoint["process_seconds"],
        "caveat": (
            "Validation only. The duration decision is based on seed 17 and is now "
            "frozen pending predeclared seed-23/42 replication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
