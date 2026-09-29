"""Audit a three-repeat same-host CPU resource record for Stage143."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/stage143-evidence/final.json"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(record: dict, *, threads: int) -> dict:
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    if (record["split"] != "validation" or record["device"] != "cpu"
            or record["precision"] != "fp32" or record["threads"] != threads
            or record["repeats"] != 3):
        raise ValueError("Wrong protocol or resource repetition count")
    if digest(ROOT / "benchmark_controls/baseline-stage3-long-s17.pt") != BASELINE_SHA:
        raise ValueError("Baseline checkpoint changed")
    if digest(ROOT / "checkpoints/stage143-openvino-order6.pt") != final["checkpoint_sha256"]:
        raise ValueError("Stage143 checkpoint changed")
    if record["baseline"]["checkpoint_sha256"] != BASELINE_SHA:
        raise ValueError("Resource baseline checkpoint changed")
    if record["candidate"]["checkpoint_sha256"] != final["checkpoint_sha256"]:
        raise ValueError("Resource candidate checkpoint changed")
    if record["baseline"]["implementation_sha256"] != digest(ROOT / "model.py"):
        raise ValueError("Baseline implementation changed")
    if (record["candidate"]["implementation_sha256"]
            != final["source_hashes"]["student_stage143_openvino_singlepass.py"]):
        raise ValueError("Stage143 implementation changed")
    for relative, expected in final["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Inference file changed: {relative}")
    assets = (ROOT / "checkpoints/stage143-openvino-order6.pt").stat().st_size
    assets += sum((ROOT / relative).stat().st_size for relative in final["inference_files"])
    if assets != final["conservative_asset_bytes"]:
        raise ValueError("Inference asset count changed")

    results = {}
    for label in ("baseline", "candidate"):
        arm = record[label]
        rows = arm["runs"]
        if len(rows) != 3 or arm["checkpoint_sha256"] != rows[0]["checkpoint_sha256"]:
            raise ValueError(f"Wrong {label} repetition identity")
        for row in rows:
            if (row["protocol"] != "7506-mp1-wt2-v2"
                    or row["split"] != "validation" or row["precision"] != "fp32"
                    or row["threads"] != threads
                    or row["targets"] != EXPECTED_TARGETS
                    or row["utf8_bytes"] != EXPECTED_BYTES
                    or row["checkpoint_sha256"] != arm["checkpoint_sha256"]
                    or row["implementation_sha256"] != arm["implementation_sha256"]
                    or row["evaluator_sha256"] != final["source_hashes"]["evaluate.py"]
                    or row["tokenizer_sha256"] != final["source_hashes"]["data/tokenizer.json"]):
                raise ValueError(f"{label} repetition protocol/coverage mismatch")
        median = statistics.median(row["seconds"] for row in rows)
        peak = max(row["peak_rss_bytes"] for row in rows)
        if abs(median - arm["median_seconds"]) > 1e-9 or peak != arm["max_peak_rss_bytes"]:
            raise ValueError(f"{label} summary arithmetic mismatch")
        results[label] = dict(median_seconds=median, max_peak_rss_bytes=peak,
                              bpb_runs=[row["bpb"] for row in rows])

    ratio = results["candidate"]["median_seconds"] / results["baseline"]["median_seconds"]
    if abs(ratio - record["candidate_to_baseline_time_ratio"]) > 1e-9:
        raise ValueError("Time-ratio arithmetic mismatch")
    if any(abs(bpb - final["validation_bpb"]) > 2e-5
           for bpb in results["candidate"]["bpb_runs"]):
        raise ValueError("Stage143 validation score mismatch")
    if max(results["candidate"]["bpb_runs"]) - min(results["candidate"]["bpb_runs"]) > 1e-9:
        raise ValueError("Stage143 repeated scores differ")

    passes = dict(cpu_time=ratio <= 5,
                  peak_rss=results["candidate"]["max_peak_rss_bytes"] <= 4 * 1024**3,
                  inference_assets=assets <= 64 * 1024**2)
    return dict(
        purpose="additional_linux_resource_observation_not_instructor_hardware_guarantee",
        protocol="7506-mp1-wt2-v2", split="validation", precision="fp32",
        requested_threads=threads, repetitions=3,
        baseline_checkpoint_sha256=BASELINE_SHA,
        candidate_checkpoint_sha256=final["checkpoint_sha256"],
        baseline=results["baseline"], candidate=results["candidate"],
        candidate_to_baseline_time_ratio=ratio,
        conservative_inference_asset_bytes=assets,
        passes=passes, all_local_gates_pass=all(passes.values()),
        no_test_scoring=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output")
    record = json.loads(args.resource.read_text(encoding="utf-8-sig"))
    result = audit(record, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_local_gates_pass"]:
        raise SystemExit("One or more Linux resource limits failed")


if __name__ == "__main__":
    main()
