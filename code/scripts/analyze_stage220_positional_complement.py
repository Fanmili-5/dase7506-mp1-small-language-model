"""Fixed validation-only positional allocation of frozen expert gains."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/stage162-evidence"
HASH143 = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
HASH155 = "0cbe8ee4aec517ba756fc5e167a95aea2b82e088b5f870e29f77337a9a499603"
TARGETS = 376_599
BYTES = 1_148_007
EXPECTED = (1.3996861620421412, 1.4098772704183504, 1.3761826641885053)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bpb(logp: np.ndarray) -> float:
    return -float(np.sum(logp, dtype=np.float64)) / math.log(2) / BYTES


def analyze() -> dict:
    a_path = REF / "stage143-target-logp.npy"
    b_path = REF / "stage155-target-logp.npy"
    if sha(a_path) != HASH143 or sha(b_path) != HASH155:
        raise ValueError("Frozen target-probability stream changed")
    a = np.load(a_path, allow_pickle=False).astype(np.float64)
    b = np.load(b_path, allow_pickle=False).astype(np.float64)
    if a.shape != b.shape or a.shape != (TARGETS,) or not (
            np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Complete finite target streams required")
    m = np.logaddexp(a, b) - math.log(2)
    scores = (bpb(a), bpb(b), bpb(m))
    if any(abs(actual - expected) > 1e-8 for actual, expected in zip(scores, EXPECTED)):
        raise ValueError(f"Frozen BPB identities changed: {scores}")
    position = np.arange(TARGETS, dtype=np.int64) % 256
    gain_nats = m - a
    total_gain_nats = float(gain_nats.sum(dtype=np.float64))
    bins = []
    for start in range(0, 256, 32):
        mask = (position >= start) & (position < start + 32)
        subtotal = float(gain_nats[mask].sum(dtype=np.float64))
        bins.append(dict(start=start, end=start + 32,
                         targets=int(mask.sum()),
                         gain_nats=subtotal,
                         gain_fraction_of_total=subtotal / total_gain_nats,
                         gain_nats_per_target=subtotal / int(mask.sum())))
    policies = {}
    for length in (32, 64, 128):
        use = position < length
        logp = np.where(use, m, a)
        policy_bpb = bpb(logp)
        policies[str(length)] = dict(
            first_positions=length, targets_with_second_expert=int(use.sum()),
            validation_bpb=policy_bpb,
            gain_vs_stage143_bpb=scores[0] - policy_bpb,
        )
    first64_share = sum(row["gain_fraction_of_total"] for row in bins[:2])
    gate = (first64_share >= 0.5
            and policies["64"]["gain_vs_stage143_bpb"] >= 0.012)
    return dict(
        purpose="validation_only_frozen_positional_complement_diagnostic",
        protocol="7506-mp1-wt2-v2", split="validation",
        targets=TARGETS, utf8_bytes=BYTES, context=256,
        source_array_sha256={"stage143": HASH143, "stage155": HASH155},
        stage143_bpb=scores[0], stage155_bpb=scores[1],
        all_positions_half_mixture_bpb=scores[2],
        total_gain_nats=total_gain_nats, bins=bins,
        fixed_first_position_policies=policies,
        first64_gain_fraction=first64_share,
        first64_gate_passed=gate,
        no_test_scoring=True,
        warning="Target probabilities are analysis artifacts; frozen two-model policies exceed inference budget",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite analysis output")
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
