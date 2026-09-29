"""Export single-copy Stage105/OpenVINO predictor and qualify all local gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup, sha
from student_stage143_openvino_singlepass import GRAPH, GRAPH_SHA256
from train_experiment import atomic_json_dump, atomic_torch_save


SOURCE_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_BPB = 1.3996861631723965
IMPLEMENTATION = "student_stage143_openvino_singlepass"
INFERENCE_FILES = (
    "student_stage143_openvino_singlepass.py",
    "student_stage105_gated_singlepass.py", "student_stage104_gated_fast.py",
    "student_stage103_gated.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "common.py", "evaluate.py", "data/tokenizer.json", "requirements.txt",
    "requirements_stage143.txt", "inference_assets/stage143-stage92-features.onnx",
)


def smoke(reference, candidate):
    inputs = (
        (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048,
        (torch.arange(512).reshape(2, 256) * 17 + 5) % 2048,
    )
    probability_error = logp_error = norm_error = 0.0
    with torch.inference_mode():
        for ids in inputs:
            expected = reference.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
            probability_error = max(
                probability_error, float((actual.exp() - expected.exp()).abs().max()))
            logp_error = max(logp_error, float((actual - expected).abs().max()))
            norm_error = max(norm_error, float(actual.logsumexp(-1).abs().max()))
        original = inputs[0][:1].clone()
        changed = original.clone()
        changed[:, 128:] = (changed[:, 128:] + 37) % 2048
        causal_error = float((candidate.predict_log_probs(original)[:, :128]
                              - candidate.predict_log_probs(changed)[:, :128])
                             .abs().max())
        pair = inputs[0]
        row_error = float((candidate.predict_log_probs(pair)[0]
                           - candidate.predict_log_probs(pair[:1])[0]).abs().max())
    result = dict(max_probability_error=probability_error,
                  max_logp_error=logp_error,
                  max_log_normalization_error=norm_error,
                  max_causal_prefix_logp_error=causal_error,
                  max_independent_row_logp_error=row_error)
    if (probability_error > 3e-6 or logp_error > 3e-4
            or norm_error > 1e-5 or causal_error > 3e-5
            or row_error > 3e-5):
        raise ValueError(f"Compact predictor parity/causality failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Refusing to overwrite Stage143 qualification")
    if (sha(args.source) != SOURCE_SHA or sha(args.baseline) != BASELINE_SHA
            or sha(GRAPH) != GRAPH_SHA256):
        raise ValueError("Unexpected frozen source/baseline/feature graph")
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage105_gated_singlepass"):
        raise ValueError("Unexpected Stage105 source payload")
    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    candidate, candidate_module_sha = make_model(IMPLEMENTATION, payload["config"], device)
    reference.load_state_dict(payload["model"], strict=True)
    candidate_keys = set(candidate.state_dict())
    source_keys = set(payload["model"])
    if not candidate_keys <= source_keys:
        raise ValueError(f"Compact model introduced untrained state: {candidate_keys - source_keys}")
    discarded = source_keys - candidate_keys
    if len(discarded) < 50 or not any(name.startswith("neural.blocks.")
                                      for name in discarded):
        raise ValueError("Feature backbone was not removed from checkpoint")
    candidate.load_state_dict({key: payload["model"][key]
                               for key in candidate_keys}, strict=True)
    reference.eval(); candidate.eval()
    smoke_result = smoke(reference, candidate)
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage143-openvino-order6.pt"
    exported = dict(payload, implementation=IMPLEMENTATION,
                    model=candidate.state_dict())
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=SOURCE_SHA, graph_sha256=GRAPH_SHA256,
        method="single_copy_fp32_openvino_features_same_frozen_head_count_gate",
        discarded_pytorch_feature_state_keys=len(discarded),
        no_new_gradient_targets=True, no_test_scoring=True, smoke=smoke_result)
    atomic_torch_save(exported, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resource_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (validation["targets"] != 376599 or validation["utf8_bytes"] != 1148007
            or validation["bpb"] >= 1.4
            or abs(validation["bpb"] - EXPECTED_BPB) > 2e-5):
        raise ValueError("Compact full validation disagrees with Stage105")
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py", "--baseline",
        str(args.baseline.resolve()), "--candidate", str(checkpoint.resolve()),
        "--repeats", "3", "--threads", "4", "--output",
        str(resource_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resource_path.read_text(encoding="utf-8"))
    asset_sizes = {name: (ROOT / name).stat().st_size for name in INFERENCE_FILES}
    assets = checkpoint.stat().st_size + sum(asset_sizes.values())
    final = dict(
        protocol=PROTOCOL, status="stage143_compact_openvino_resource_audited",
        source_checkpoint_sha256=SOURCE_SHA, checkpoint_sha256=sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        implementation_sha256=candidate_module_sha,
        graph_sha256=GRAPH_SHA256, validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        baseline_median_seconds=resources["baseline"]["median_seconds"],
        candidate_median_seconds=resources["candidate"]["median_seconds"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets, asset_sizes=asset_sizes,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        smoke=smoke_result, inference_files=list(INFERENCE_FILES),
        source_hashes={name: sha(ROOT / name) for name in INFERENCE_FILES},
        split="validation", precision="fp32", no_test_scoring=True,
    )
    final["qualified"] = all((final["validation_bpb"] < 1.4,
                              final["within_five_x_time_limit"],
                              final["within_four_gib_peak_rss_limit"],
                              final["within_64mib_asset_limit"]))
    atomic_json_dump(final, args.run_dir / "final.json")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
