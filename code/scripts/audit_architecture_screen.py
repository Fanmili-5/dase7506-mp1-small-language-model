"""Audit a completed Stage-14 screen, including real weights and ancestry.

Run on the Windows artifact directory or a complete copy after the job finishes.
Refuses unfinished screens; never evaluates a model or loads benchmark text.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from common import PROTOCOL, sha

NAMES = {"A-wide", "B-copy", "C-mos"}
AVERAGE_STEPS = (6000, 6300, 6600, 6900, 7200)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_score(row, checkpoint_hash, implementation_hash, sources):
    expected = (PROTOCOL, "validation", "cpu", "fp32", 376599, 1148007)
    actual = tuple(row[key] for key in ("protocol", "split", "device", "precision", "targets", "utf8_bytes"))
    if actual != expected:
        raise ValueError("Score protocol, coverage or precision mismatch")
    if row["checkpoint_sha256"] != checkpoint_hash or row["implementation_sha256"] != implementation_hash:
        raise ValueError("Score belongs to a different checkpoint or implementation")
    if row["evaluator_sha256"] != sources["evaluate.py"] or row["tokenizer_sha256"] != sources["data/tokenizer.json"]:
        raise ValueError("Evaluator or tokenizer hash mismatch")
    bpb = row["bpb"]
    if not math.isfinite(bpb) or bpb < 0 or not math.isclose(
        bpb, row["nll_nats"] / math.log(2) / 1148007, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("BPB is not consistent with total NLL")
    return bpb


def check_resource(result, candidate_hash, baseline_hash, implementation_hash, sources, expected_bpb=None):
    if tuple(result[k] for k in ("split", "device", "precision", "threads", "repeats")) != (
        "validation", "cpu", "fp32", 4, 3
    ):
        raise ValueError("Expected three fresh four-thread CPU FP32 resource repetitions")
    summaries = {}
    for label, checkpoint_hash in (("baseline", baseline_hash), ("candidate", candidate_hash)):
        group = result[label]
        if group["checkpoint_sha256"] != checkpoint_hash or len(group["runs"]) != 3:
            raise ValueError("Resource checkpoint/repetition mismatch")
        seconds, peaks = [], []
        for row in group["runs"]:
            module_hash = implementation_hash if label == "candidate" else sources["model.py"]
            # The original CPU-only probe records device at benchmark level;
            # individual rows may omit it. An explicit conflicting device fails.
            bpb = check_score({"device": result["device"], **row}, checkpoint_hash, module_hash, sources)
            if label == "candidate" and expected_bpb is not None and abs(bpb - expected_bpb) > 1e-6:
                raise ValueError("Resource scoring differs from the selected checkpoint score")
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0 or row["peak_rss_bytes"] <= 0:
                raise ValueError("Invalid timing or peak memory")
            seconds.append(row["seconds"])
            peaks.append(row["peak_rss_bytes"])
        median, peak = statistics.median(seconds), max(peaks)
        if not math.isclose(median, group["median_seconds"]) or peak != group["max_peak_rss_bytes"]:
            raise ValueError("Resource aggregate does not match the raw repetitions")
        summaries[label] = (median, peak)
    ratio = summaries["candidate"][0] / summaries["baseline"][0]
    peak = summaries["candidate"][1]
    if not math.isclose(ratio, result["candidate_to_baseline_time_ratio"]):
        raise ValueError("Incorrect reported time ratio")
    time_pass, ram_pass = ratio <= 5, peak <= 4 * 1024**3
    if (result["within_five_x_time_limit"], result["within_four_gib_peak_rss_limit"]) != (time_pass, ram_pass):
        raise ValueError("Incorrect resource pass flags")
    return {"time_ratio": ratio, "peak_rss_bytes": peak, "time_pass": time_pass, "ram_pass": ram_pass}


def audit(directory):
    screen = read(directory / "screen.json")
    if screen["status"] != "completed":
        raise ValueError("Screen is unfinished; do not select among partial results")
    rows = screen["candidates"]
    if len(rows) != 3 or {row["name"] for row in rows} != NAMES:
        raise ValueError("Expected all three preregistered candidates")
    sources = dict(screen["source_hashes"])
    for relative, expected in sources.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    manifest = read(ROOT / "PACKAGE_MANIFEST.json")
    sources["model.py"] = manifest["code/model.py"]
    if sha(ROOT / "model.py") != sources["model.py"]:
        raise ValueError("Official baseline source changed")
    control = read(ROOT / "results/stage12_summary.json")
    if sha(ROOT / "results/stage12_summary.json") != screen["control_summary_sha256"]:
        raise ValueError("Control summary changed")
    endpoint_control = next(x["validation_bpb"] for x in control["candidates"] if x["checkpoint"] == "checkpoint.pt")
    average_control = control["selected"]["validation_bpb"]
    results = []
    for row in rows:
        name = row["name"]
        implementation_hash = sources[f"{row['implementation']}.py"]
        probe = directory / "preflight" / f"{name}.pt"
        resource = check_resource(read(directory / "preflight" / f"{name}-resource.json"),
                                  sha(probe), screen["baseline_sha256"], implementation_hash, sources)
        inference_paths = [ROOT / p for p in ("student.py", "student_structured.py", "common.py",
                                               "evaluate.py", "data/tokenizer.json", "requirements.txt")]
        auxiliary_bytes = sum(p.stat().st_size for p in inference_paths)
        passed = resource["time_pass"] and resource["ram_pass"] and probe.stat().st_size + auxiliary_bytes <= 64 * 1024**2
        if passed != row["preflight_pass"]:
            raise ValueError(f"Preflight decision mismatch: {name}")
        if not passed:
            if row["status"] != "resource_rejected" or (directory / name).exists():
                raise ValueError("A resource-rejected candidate unexpectedly trained")
            results.append({"name": name, "status": "resource_rejected", "preflight": resource})
            continue
        run = directory / name
        metrics = read(run / "metrics.json")
        plan = metrics["plan"]
        for key, value in {"steps": 7200, "seed": 17, "micro_batch_size": 32, "grad_accum": 1,
                           "schedule": "baseline", "eval_every": 300, "keep_eval_checkpoints": True,
                           "learning_rate": 0.001, "min_lr_ratio": 0.1, "weight_decay": 0.1,
                           "warmup_steps": 100, "betas": [0.9, 0.999], "grad_clip": 1.0}.items():
            if plan[key] != value:
                raise ValueError(f"Training recipe differs at {name}: {key}")
        if metrics["train_tokens"] != 58982400 or metrics["trainer_sha256"] != sources["train_experiment.py"]:
            raise ValueError("Training targets or trainer hash mismatch")
        config = read(ROOT / "configs" / row["config"])
        if metrics["config_sha256"] != sha(ROOT / "configs" / row["config"]) or plan["config"] != config:
            raise ValueError("Training configuration differs from preregistration")
        result = {"name": name, "status": "quality_scored", "preflight": resource,
                  "train_targets": metrics["train_tokens"], "train_seconds": metrics["train_seconds"]}
        for label, filename in (("endpoint", "checkpoint.pt"), ("average", "average-last5.pt")):
            checkpoint = run / filename
            digest = sha(checkpoint)
            score = read(run / f"{label}-validation-cpu-fp32.json")
            bpb = check_score(score, digest, implementation_hash, sources)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (payload["protocol"], payload["implementation"], payload["config"], payload["seed"], payload["train_tokens"]) != (
                PROTOCOL, row["implementation"], config, 17, 58982400
            ):
                raise ValueError("Checkpoint metadata differs from its recipe")
            if label == "average":
                ancestry = payload["averaging_ancestry"]
                snapshots = [run / "checkpoints" / f"step-{step:06d}.pt" for step in AVERAGE_STEPS]
                if (ancestry["method"] != "uniform_same_trajectory_parameter_average"
                    or ancestry["source_count"] != 5
                    or ancestry["source_train_targets"] != [s * 8192 for s in AVERAGE_STEPS]
                    or ancestry["source_checkpoint_sha256"] != [sha(p) for p in snapshots]):
                    raise ValueError("Averaging ancestry differs from the fixed last-five rule")
            result[f"{label}_bpb"] = bpb
            result[f"{label}_checkpoint_sha256"] = digest
            if not math.isclose(bpb, row[f"{label}_bpb"], abs_tol=1e-10):
                raise ValueError("Screen score does not match raw evaluation evidence")
        result["endpoint_gain"] = endpoint_control - result["endpoint_bpb"]
        result["average_gain"] = average_control - result["average_bpb"]
        result["inference_asset_bytes"] = (run / "average-last5.pt").stat().st_size + auxiliary_bytes
        result["eligible_for_followup"] = False
        if result["average_gain"] >= 0.003:
            final = check_resource(read(run / "final-resource.json"), result["average_checkpoint_sha256"],
                                   screen["baseline_sha256"], implementation_hash, sources, result["average_bpb"])
            result["final_resource"] = final
            result["eligible_for_followup"] = final["time_pass"] and final["ram_pass"] and result["inference_asset_bytes"] <= 64 * 1024**2
        results.append(result)
    candidates = [r for r in results if r.get("eligible_for_followup")]
    return {"protocol": PROTOCOL, "split": "validation", "precision": "fp32", "device": "cpu",
            "audit_script_sha256": sha(Path(__file__)),
            "screen_sha256": sha(directory / "screen.json"), "endpoint_control_bpb": endpoint_control,
            "average_control_bpb": average_control, "candidates": results,
            "new_training_targets": sum(r.get("train_targets", 0) for r in results),
            "followup_candidate": min(candidates, key=lambda r: r["average_bpb"])["name"] if candidates else None,
            "status": "audited single-seed screen; not a frozen submission; no new test result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
