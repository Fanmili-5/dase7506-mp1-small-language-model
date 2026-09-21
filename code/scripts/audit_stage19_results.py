"""Audit completed Stage19 evidence and local exact checkpoint copies.

Checks score arithmetic, coverage, receipt/source hashes, mixture groups, and
resource aggregation. Does not claim to reproduce training or checkpoint averages.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, sha

H_SHA = "2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7"
BASE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
COUNTS_SHA = "b1898559c73ccf62e6e945c1a9269e6fe2b0ee4499b5a1d88e7e30020c194230"


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"Mismatch: {a} vs {b}")


def score(row, expected_sha):
    assert row["protocol"] == PROTOCOL and row["split"] == "validation"
    assert row["precision"] == "fp32" and row["checkpoint_sha256"] == expected_sha
    assert row["targets"] == 376599 and row["utf8_bytes"] == 1148007
    close(row["bpb"], row["nll_nats"] / math.log(2) / row["utf8_bytes"])
    for name, key in (("evaluate.py", "evaluator_sha256"), ("data/tokenizer.json", "tokenizer_sha256")):
        assert sha(ROOT / name) == row[key]


def resource(report, expected_sha, expected_bpb):
    assert report["repeats"] == 3 and report["threads"] == 4
    assert report["device"] == "cpu" and report["split"] == "validation" and report["precision"] == "fp32"
    for label, digest in (("baseline", BASE_SHA), ("candidate", expected_sha)):
        section = report[label]
        assert len(section["runs"]) == 3 and section["checkpoint_sha256"] == digest
        for row in section["runs"]:
            score(row, digest)
            if label == "candidate":
                close(row["bpb"], expected_bpb)
        close(section["median_seconds"], statistics.median(r["seconds"] for r in section["runs"]))
        assert section["max_peak_rss_bytes"] == max(r["peak_rss_bytes"] for r in section["runs"])
    ratio = report["candidate"]["median_seconds"] / report["baseline"]["median_seconds"]
    close(report["candidate_to_baseline_time_ratio"], ratio)
    memory = report["candidate"]["max_peak_rss_bytes"]
    assert report["within_five_x_time_limit"] == (ratio <= 5)
    assert report["within_four_gib_peak_rss_limit"] == (memory <= 4 * 1024**3)
    return dict(cpu_ratio=ratio, peak_rss_bytes=memory, cpu_pass=ratio <= 5, ram_pass=memory <= 4 * 1024**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--neural", type=Path, required=True)
    p.add_argument("--hybrid", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new audit output")
    def read(name):
        return json.loads((args.evidence / name).read_text(encoding="utf-8-sig"))
    scan = read("mixture/scan.json")
    assert sha(args.neural) == H_SHA and scan["reference_sha256"] == H_SHA
    assert scan["counts_sha256"] == COUNTS_SHA
    assert scan["targets"] == 376599 and scan["utf8_bytes"] == 1148007
    assert [r["weight"] for r in scan["candidates"]] == [0., .05, .1]
    assert scan["status"].startswith("completed validation")
    for name, digest in scan["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    for row in scan["candidates"]:
        close(row["bpb"], row["nll_nats"] / math.log(2) / scan["utf8_bytes"])
        groups = scan["diagnostic_groups"][str(row["weight"])]
        assert sum(g["targets"] for g in groups.values()) == 376599
        close(sum(g["nll_nats"] for g in groups.values()), row["nll_nats"])
    best = min(scan["candidates"], key=lambda r: r["bpb"])
    assert best == scan["provisional_best"] and best["weight"] > 0
    hybrid_sha = sha(args.hybrid)
    assert scan["best_checkpoint_sha256"] == hybrid_sha
    official = read("mixture/best-validation-cpu-fp32.json")
    score(official, hybrid_sha)
    close(official["bpb"], best["bpb"])
    files = ("student.py", "student_structured.py", "common.py", "evaluate.py", "requirements.txt", "data/tokenizer.json")
    core = sum((ROOT / f).stat().st_size for f in files)
    models = {}
    for label, path, digest, bpb, result_file in (
        ("H", args.neural, H_SHA, scan["candidates"][0]["bpb"], "h-resources.json"),
        ("hybrid", args.hybrid, hybrid_sha, official["bpb"], "mixture-resources.json"),
    ):
        result = resource(read(result_file), digest, bpb)
        assets = core + path.stat().st_size + ((ROOT / "student_ngram.py").stat().st_size if label == "hybrid" else 0)
        if label == "hybrid":
            assert assets == scan["inference_asset_bytes"]
        result.update(bpb=bpb, checkpoint_sha256=digest, asset_bytes=assets, asset_pass=assets <= 64 * 1024**2)
        result["qualified_on_measured_windows_cpu"] = result["cpu_pass"] and result["ram_pass"] and result["asset_pass"]
        models[label] = result
    profile = read("profile.json")
    assert profile["checkpoint_sha256"] == H_SHA
    assert profile["profiler_sha256"] == sha(ROOT / "scripts/profile_structured_cpu.py")
    result = dict(status="receipts_and_resource_aggregation_verified", protocol=PROTOCOL,
        split="validation", models=models, new_gradient_targets=0,
        file_hashes={str(f.relative_to(args.evidence)): sha(f) for f in sorted(args.evidence.rglob("*.json"))},
        limitations="Does not rerun training or reaverage weights. Microbenchmarks are diagnostic, not resource gates. Qualification applies to measured Windows CPU; no test score or universal timing guarantee.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result["models"], indent=2))


if __name__ == "__main__":
    main()
