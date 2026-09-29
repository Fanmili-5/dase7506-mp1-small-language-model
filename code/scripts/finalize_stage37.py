"""Consolidate the exact Stage37 validation and resource gates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, sha
from train_experiment import atomic_json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.run_dir / "calibrated-collapsed.pt"
    equivalence = json.loads((args.run_dir / "equivalence.json").read_text())
    validation = json.loads((args.run_dir / "validation-cpu-fp32.json").read_text())
    resources = json.loads((args.run_dir / "resources.json").read_text())
    digest = sha(checkpoint)
    if (equivalence["protocol"] != PROTOCOL or validation["protocol"] != PROTOCOL
            or digest != equivalence["checkpoint_sha256"]
            or digest != validation["checkpoint_sha256"]
            or digest != resources["candidate"]["checkpoint_sha256"]
            or abs(validation["bpb"] - equivalence["optimized_bpb"]) > 1e-6):
        raise ValueError("Stage37 hash, protocol, or score mismatch")
    result = dict(
        protocol=PROTOCOL, checkpoint_sha256=digest, validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=equivalence["conservative_asset_bytes"],
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=equivalence["conservative_asset_bytes"] <= 64 * 1024 ** 2,
        split="validation", precision="fp32", no_test_scoring=True,
    )
    result["qualified"] = all((result["within_five_x_time_limit"],
                                result["within_four_gib_peak_rss_limit"],
                                result["within_64mib_asset_limit"]))
    atomic_json_dump(result, args.run_dir / "qualification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
