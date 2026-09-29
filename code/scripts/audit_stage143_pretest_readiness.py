"""Recheck Stage143 evidence and assets without freezing or scoring test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/stage143-evidence"
CHECKPOINT = ROOT / "checkpoints/stage143-openvino-order6.pt"
MAX_ASSETS = 64 * 1024**2
MAX_RAM = 4 * 1024**3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit() -> dict:
    final = read_json(EVIDENCE / "final.json")
    windows = read_json(EVIDENCE / "validation-cpu-fp32.json")
    clean = read_json(EVIDENCE / "clean-extract-validation.json")
    linux = read_json(ROOT / "results/stage166-linux-evidence/stage143-linux-x86-validation.json")
    resources = read_json(EVIDENCE / "resources.json")

    checkpoint_sha = digest(CHECKPOINT)
    if checkpoint_sha != final["checkpoint_sha256"]:
        raise ValueError("Stage143 checkpoint has changed")
    if checkpoint_sha != resources["candidate"]["checkpoint_sha256"]:
        raise ValueError("Resource run used a different checkpoint")
    if resources["repeats"] != 3 or resources["threads"] != 4:
        raise ValueError("Missing three four-thread resource repetitions")
    if resources["split"] != "validation" or resources["precision"] != "fp32":
        raise ValueError("Resource protocol is not validation CPU FP32")
    if resources["device"] != "cpu":
        raise ValueError("Resource protocol is not CPU")
    if set(final["inference_files"]) != set(final["source_hashes"]):
        raise ValueError("The counted inference file manifest is incomplete")

    asset_sizes = {}
    for relative, expected in final["source_hashes"].items():
        path = ROOT / relative
        if digest(path) != expected:
            raise ValueError(f"Inference source/asset changed: {relative}")
        size = path.stat().st_size
        if size != final["asset_sizes"][relative]:
            raise ValueError(f"Inference asset byte count changed: {relative}")
        asset_sizes[relative] = size
    counted_assets = CHECKPOINT.stat().st_size + sum(asset_sizes.values())
    if counted_assets != final["conservative_asset_bytes"] or counted_assets > MAX_ASSETS:
        raise ValueError("Inference asset limit/count mismatch")
    if final["source_hashes"]["inference_assets/stage143-stage92-features.onnx"] != final["graph_sha256"]:
        raise ValueError("The graph hash is not included in the counted assets")

    expected_targets, expected_bytes = 376_599, 1_148_007
    for name, record in (("Windows", windows), ("clean extraction", clean),
                         ("Linux", linux)):
        if (record["protocol"] != "7506-mp1-wt2-v2"
                or record["split"] != "validation" or record["device"] != "cpu"
                or record["precision"] != "fp32"
                or record["targets"] != expected_targets
                or record["utf8_bytes"] != expected_bytes
                or record["checkpoint_sha256"] != checkpoint_sha
                or record["implementation_sha256"] != final["source_hashes"]["student_stage143_openvino_singlepass.py"]
                or record["evaluator_sha256"] != final["source_hashes"]["evaluate.py"]
                or record["tokenizer_sha256"] != final["source_hashes"]["data/tokenizer.json"]):
            raise ValueError(f"{name} validation identity/coverage mismatch")
    if abs(windows["bpb"] - clean["bpb"]) > 2e-5 or abs(windows["bpb"] - linux["bpb"]) > 2e-5:
        raise ValueError("Independent validation scores differ materially")
    if abs(windows["bpb"] - final["validation_bpb"]) > 1e-10:
        raise ValueError("The qualified score does not match the Windows validation")

    baseline_times = [row["seconds"] for row in resources["baseline"]["runs"]]
    candidate_times = [row["seconds"] for row in resources["candidate"]["runs"]]
    if len(baseline_times) != 3 or len(candidate_times) != 3:
        raise ValueError("Resource run count mismatch")
    ratio = statistics.median(candidate_times) / statistics.median(baseline_times)
    peak_rss = max(row["peak_rss_bytes"] for row in resources["candidate"]["runs"])
    if (abs(ratio - resources["candidate_to_baseline_time_ratio"]) > 1e-9
            or peak_rss != resources["candidate"]["max_peak_rss_bytes"]
            or ratio > 5 or peak_rss > MAX_RAM):
        raise ValueError("Resource evidence/limit mismatch")
    for name in ("baseline", "candidate"):
        for row in resources[name]["runs"]:
            if (row["protocol"] != "7506-mp1-wt2-v2"
                    or row["precision"] != "fp32" or row["split"] != "validation"
                    or row["checkpoint_sha256"] != resources[name]["checkpoint_sha256"]
                    or row["targets"] != expected_targets
                    or row["utf8_bytes"] != expected_bytes):
                raise ValueError("Resource run protocol/coverage mismatch")
            if name == "candidate" and abs(row["bpb"] - windows["bpb"]) > 1e-10:
                raise ValueError("Resource candidate scored a different model")

    return {
        "status": "pretest_readiness_audited_only_not_frozen_or_submitted",
        "checkpoint_sha256": checkpoint_sha,
        "graph_sha256": final["graph_sha256"],
        "windows_validation_bpb": windows["bpb"],
        "clean_extract_validation_bpb": clean["bpb"],
        "linux_validation_bpb": linux["bpb"],
        "validation_targets": expected_targets,
        "validation_utf8_bytes": expected_bytes,
        "windows_cpu_time_ratio": ratio,
        "windows_peak_rss_bytes": peak_rss,
        "conservative_inference_asset_bytes": counted_assets,
        "inference_file_count_plus_checkpoint": len(asset_sizes) + 1,
        "pending": [
            "explicit method freeze and immutable source/checkpoint manifest",
            "one full-test FP32 CPU score only after freeze",
            "matching final report PDF of at most 10 pages",
            "student ID and course website score issue before 29 September",
            "publicly accessible immutable code and matching checkpoint-bundle links by 30 September",
        ],
    }


def main() -> None:
    result = audit()
    output = EVIDENCE / "pretest-readiness.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
