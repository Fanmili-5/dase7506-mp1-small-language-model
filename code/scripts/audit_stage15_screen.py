"""Audit completed Stage15 raw scores/resources and recompute weight averages.

Read-only on experiment artifacts: no training or dataset scoring. An unfinished
screen is not selectable. JSON output is written only after every check passes.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import PROTOCOL, sha
from scripts.audit_architecture_screen import check_score, check_resource, read
from scripts.average_checkpoints import average_checkpoints
from scripts.run_stage15_screen import SPECS, average_steps, asset_total
from train_experiment import atomic_json_dump


def check_average(run, payload, steps):
    paths = [run / "checkpoints" / f"step-{step:06d}.pt" for step in average_steps(steps)]
    expected_targets = [step * 8192 for step in average_steps(steps)]
    ancestry = payload["averaging_ancestry"]
    if ancestry["source_train_targets"] != expected_targets:
        raise ValueError("Average uses unexpected snapshot target counts")
    rebuilt = average_checkpoints(paths)
    if rebuilt["averaging_ancestry"] != ancestry:
        raise ValueError("Averaging ancestry does not match source files")
    if rebuilt["model"].keys() != payload["model"].keys():
        raise ValueError("Average model state keys differ")
    for name, tensor in rebuilt["model"].items():
        if not torch.equal(tensor, payload["model"][name]):
            raise ValueError(f"Recomputed average differs: {name}")


def audit(directory):
    screen = read(directory / "screen.json")
    if (screen["status"], screen["protocol"], screen["split"]) != ("completed", PROTOCOL, "validation"):
        raise ValueError("Stage15 must be completed under the validation protocol")
    rows = screen["candidates"]
    if len(rows) != len(SPECS) or [r["name"] for r in rows] != [s[0] for s in SPECS]:
        raise ValueError("Missing or reordered preregistered candidates")
    sources = dict(screen["source_hashes"])
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed pinned source: {name}")
    sources["model.py"] = read(ROOT / "PACKAGE_MANIFEST.json")["code/model.py"]
    if sha(ROOT / "model.py") != sources["model.py"]:
        raise ValueError("Baseline source changed")
    torch.set_num_threads(4)
    results = []
    for row, (name, impl, config_file, steps) in zip(rows, SPECS):
        if (row["implementation"], row["config"], row["steps"], row["planned_targets"], row["average_steps"]) != (
                impl, config_file, steps, steps * 8192, list(average_steps(steps))):
            raise ValueError(f"Unexpected declared recipe: {name}")
        implementation_sha = sources[f"{impl}.py"]
        probe = directory / "preflight" / f"{name}.pt"
        resource = check_resource(read(directory / "preflight" / f"{name}-resource.json"),
            sha(probe), screen["baseline_sha256"], implementation_sha, sources)
        eligible = resource["time_pass"] and resource["ram_pass"] and asset_total(probe) <= 64 * 1024**2
        if eligible != row["preflight_pass"]:
            raise ValueError("Preflight eligibility mismatch")
        run = directory / name
        if not eligible:
            if row["status"] != "resource_rejected" or run.exists():
                raise ValueError("Rejected candidate unexpectedly trained")
            results.append(dict(name=name, status="resource_rejected", preflight=resource))
            continue
        if row["status"] != "completed":
            raise ValueError(f"Incomplete candidate: {name}")
        metrics = read(run / "metrics.json")
        config = read(ROOT / "configs" / config_file)
        expected = dict(steps=steps, seed=17, micro_batch_size=32, grad_accum=1,
            schedule="baseline", eval_every=300, keep_eval_checkpoints=True,
            learning_rate=.001, min_lr_ratio=.1, weight_decay=.1, warmup_steps=100,
            betas=[.9, .999], grad_clip=1., config=config)
        for key, value in expected.items():
            if metrics["plan"][key] != value:
                raise ValueError(f"Training recipe mismatch: {name}/{key}")
        if (metrics["train_tokens"] != steps * 8192 or
                metrics["trainer_sha256"] != sources["train_experiment.py"] or
                metrics["implementation_sha256"] != implementation_sha or
                metrics["config_sha256"] != sha(ROOT / "configs" / config_file)):
            raise ValueError("Trainer/config/implementation/budget provenance mismatch")
        result = dict(name=name, train_targets=metrics["train_tokens"], train_seconds=metrics["train_seconds"],
                      preflight=resource, eligible_for_followup=False)
        for label, filename in (("endpoint", "checkpoint.pt"), ("average", "average-last5.pt")):
            checkpoint = run / filename
            digest = sha(checkpoint)
            score = read(run / f"{label}-validation-cpu-fp32.json")
            bpb = check_score(score, digest, implementation_sha, sources)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (payload["protocol"], payload["implementation"], payload["config"], payload["seed"], payload["train_tokens"]) != (
                    PROTOCOL, impl, config, 17, steps * 8192):
                raise ValueError("Checkpoint metadata mismatch")
            if label == "average":
                check_average(run, payload, steps)
            elif metrics["checkpoint_sha256"] != digest:
                raise ValueError("Training metrics endpoint checkpoint mismatch")
            if row[f"{label}_checkpoint_sha256"] != digest or not math.isclose(bpb, row[f"{label}_bpb"], abs_tol=1e-10):
                raise ValueError("Screen summary differs from raw checkpoint score")
            result[f"{label}_bpb"], result[f"{label}_checkpoint_sha256"] = bpb, digest
        checkpoint = run / "average-last5.pt"
        result["asset_bytes"] = asset_total(checkpoint)
        if result["asset_bytes"] != row["final_asset_bytes"]:
            raise ValueError("Inference asset accounting differs")
        if screen["reference_average_bpb"] - result["average_bpb"] >= .003:
            measured = check_resource(read(run / "final-resource.json"), sha(checkpoint),
                screen["baseline_sha256"], implementation_sha, sources, result["average_bpb"])
            passed = measured["time_pass"] and measured["ram_pass"] and result["asset_bytes"] <= 64 * 1024**2
            if row["final_resource_pass"] != passed:
                raise ValueError("Final resource decision mismatch")
            result.update(final_resource=measured, eligible_for_followup=passed)
        results.append(result)
    targets = sum(r.get("train_targets", 0) for r in results)
    if targets != screen["actual_new_training_targets"]:
        raise ValueError("Total search cost mismatch")
    qualified = [r for r in results if r.get("eligible_for_followup")]
    return dict(protocol=PROTOCOL, split="validation", screen_sha256=sha(directory / "screen.json"),
        audit_script_sha256=sha(Path(__file__)), candidates=results, new_training_targets=targets,
        followup_candidate=min(qualified, key=lambda r: r["average_bpb"])["name"] if qualified else "B-copy-reference",
        status="audited raw scores, resources and exact averages; single seed; no test result")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run_dir)
    # Repeated audits are allowed only when evidence is byte-for-byte identical.
    if args.output.exists():
        if read(args.output) != result:
            raise FileExistsError("Existing audit differs; choose a new output")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
