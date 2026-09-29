"""Fixed validation-only frozen-expert mixtures; never a deployable model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from common import PROTOCOL, sha


STAGE143_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
STAGE155_SHA = "0cbe8ee4aec517ba756fc5e167a95aea2b82e088b5f870e29f77337a9a499603"
EXPECTED_BPB = (1.3996861620421412, 1.4098772704150238, 1.3816233893)
RAW_BYTES = 1_148_007
TARGETS = 376_599
CELLS = {
    "A": (0.50, 0.25, 0.25),
    "B": (1 / 3, 1 / 3, 1 / 3),
    "C": (0.25, 0.25, 0.50),
    "D": (0.50, 0.00, 0.50),
    "E": (0.00, 0.50, 0.50),
}


def bpb(logp: np.ndarray) -> float:
    return -float(logp.sum()) / math.log(2) / RAW_BYTES


def mix_bpb(logps: tuple[np.ndarray, ...], weights: tuple[float, ...]) -> float:
    if len(logps) != len(weights) or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("Fixed convex-mixture weights are invalid")
    probability = sum(weight * np.exp(logp) for weight, logp in zip(weights, logps))
    if not np.isfinite(probability).all() or (probability <= 0).any():
        raise ValueError("Invalid target probabilities")
    return bpb(np.log(probability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage143", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--stage91", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new Stage218 result path")
    inputs = (args.stage143, args.stage155, args.stage91)
    meta_path = args.stage91.with_suffix(".json")
    stage91_meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    if stage91_meta.get("protocol") != PROTOCOL or stage91_meta.get("split") != "validation":
        raise ValueError("Stage91 array metadata is not validation evidence")
    hashes = tuple(sha(path) for path in inputs)
    if hashes != (STAGE143_SHA, STAGE155_SHA, stage91_meta["array_sha256"]):
        raise ValueError("Frozen target array changed")
    logps = tuple(np.load(path, allow_pickle=False).astype(np.float64) for path in inputs)
    if any(array.shape != (TARGETS,) or not np.isfinite(array).all()
           for array in logps):
        raise ValueError("Incomplete or invalid frozen target stream")
    individual = tuple(bpb(array) for array in logps)
    if any(abs(observed - expected) > 2e-5
           for observed, expected in zip(individual, EXPECTED_BPB)):
        raise ValueError("Frozen model validation reproduction failed")
    half = mix_bpb(logps, (0.5, 0.5, 0.0))
    if abs(half - 1.3761826641885053) > 2e-5:
        raise ValueError("Stage143/155 half-mixture reproduction failed")
    rows = [
        {"cell": name, "stage143_weight": weights[0],
         "stage155_weight": weights[1], "stage91_weight": weights[2],
         "bpb": mix_bpb(logps, weights)}
        for name, weights in CELLS.items()
    ]
    best = min(rows, key=lambda row: row["bpb"])
    result = {
        "purpose": "validation_only_over_budget_frozen_expert_ceiling_not_inference",
        "protocol": PROTOCOL, "split": "validation", "no_test_scoring": True,
        "targets": TARGETS, "utf8_bytes": RAW_BYTES,
        "input_sha256": dict(zip(("stage143", "stage155", "stage91"), hashes)),
        "stage91_metadata_sha256": sha(meta_path),
        "individual_bpb": dict(zip(("stage143", "stage155", "stage91"), individual)),
        "stage143_stage155_half_bpb": half,
        "fixed_mixture_results": rows, "best_fixed_mixture": best,
        "fixed_mixture_reaches_student_validation_target": best["bpb"] < 1.35,
        "models_combined_exceed_course_cpu_asset_budget": True,
        "target_probability_arrays_are_label_aware_diagnostics": True,
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(ROOT / "docs/STAGE218_FROZEN_EXPERT_CEILING_PLAN_20260928.md"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
