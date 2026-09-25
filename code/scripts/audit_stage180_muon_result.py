"""Audit the prespecified Stage180 pilot gate using complete validation metrics."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import sha
from train_experiment import atomic_json_dump

BASELINE_METRICS = ROOT / "results/stage54-evidence/metrics.json"
EXPECTED_STEPS = list(range(300, 2401, 300))
EXPECTED_TARGETS = 376599
EXPECTED_BYTES = 1148007


def audit(metrics_path: Path) -> dict:
    pilot = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    control = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    control_step = next(row for row in control["validation_history"] if row["step"] == 2400)
    assert pilot["status"] == "completed_pilot_validation_only"
    assert pilot["no_test_scoring"] is True
    assert pilot["seed"] == control["seed"] == 17
    assert pilot["parameters"] == control["parameters"]
    assert pilot["steps"] == 2400 and pilot["learning_rate_horizon"] == 7200
    assert pilot["primary_targets"] == 2400 * 32 * 256
    assert pilot["clamped_starts"] >= 0
    assert [row["step"] for row in pilot["validation_history"]] == EXPECTED_STEPS
    assert all(row["targets"] == EXPECTED_TARGETS and
               row["utf8_bytes"] == EXPECTED_BYTES
               for row in pilot["validation_history"])
    assert abs(control_step["bpb"] - pilot["prespecified_baseline_step_2400_bpb"]) < 1e-8
    threshold = control_step["bpb"] - 0.020
    assert abs(threshold - pilot["prespecified_continue_threshold_bpb"]) < 1e-8
    final = pilot["validation_history"][-1]
    assert final == pilot["final_validation"]
    assert bool(final["bpb"] <= threshold) == pilot["continue_by_bpb"]
    assert abs(final["bpb"] - control_step["bpb"] - pilot["margin_vs_stage54_step2400"]) < 1e-8
    for name, digest in pilot["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    assert set(pilot["muon_parameter_names"]).isdisjoint(pilot["adamw_parameter_names"])
    assert "token.weight" in pilot["adamw_parameter_names"]
    assert "token.weight" not in pilot["muon_parameter_names"]
    return dict(status="audited_validation_only", pilot_step2400_bpb=final["bpb"],
                stage54_step2400_bpb=control_step["bpb"],
                margin_bpb=final["bpb"] - control_step["bpb"],
                prespecified_threshold_bpb=threshold,
                continue_by_bpb=pilot["continue_by_bpb"],
                train_seconds=pilot["train_seconds"],
                stage54_train_seconds_step2400=next(
                    row["train_seconds"] for row in control["history"] if row["step"] == 2400),
                no_test_scoring=True, source_hashes_verified=len(pilot["source_hashes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.metrics)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
