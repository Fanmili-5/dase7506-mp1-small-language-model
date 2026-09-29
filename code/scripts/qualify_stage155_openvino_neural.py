"""Export Stage155 single-copy FP32 graph and audit complete CPU gates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch

from common import PROTOCOL, make_model, setup, sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper, make_portable
from student_hybrid_conv_rdrop import inference_config, inference_state
from student_stage155_openvino_neural import GRAPH
from train_experiment import atomic_json_dump, atomic_torch_save


BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
STAGE143_BPB = 1.399686162042141
IMPLEMENTATION = "student_stage155_openvino_neural"
INFERENCE_FILES = (
    "student_stage155_openvino_neural.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt", "requirements_stage143.txt",
    "inference_assets/stage155-full-features.onnx",
)


def smoke(reference, candidate):
    cases = (
        (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048,
        (torch.arange(512).reshape(2, 256) * 17 + 5) % 2048,
    )
    probability_error = logp_error = norm_error = 0.0
    with torch.inference_mode():
        for ids in cases:
            expected = reference.predict_log_probs(ids)
            actual = candidate.predict_log_probs(ids)
            probability_error = max(probability_error,
                                    float((actual.exp() - expected.exp()).abs().max()))
            logp_error = max(logp_error, float((actual - expected).abs().max()))
            norm_error = max(norm_error, float(actual.logsumexp(-1).abs().max()))
        original = cases[0][:1].clone()
        changed = original.clone()
        changed[:, 128:] = (changed[:, 128:] + 37) % 2048
        causal_error = float((candidate.predict_log_probs(original)[:, :128]
                              - candidate.predict_log_probs(changed)[:, :128])
                             .abs().max())
        row_error = float((candidate.predict_log_probs(cases[0])[0]
                           - candidate.predict_log_probs(cases[0][:1])[0]).abs().max())
    result = dict(max_probability_error=probability_error,
                  max_logp_error=logp_error,
                  max_log_normalization_error=norm_error,
                  max_causal_prefix_logp_error=causal_error,
                  max_independent_row_logp_error=row_error)
    if (probability_error > 3e-6 or logp_error > 3e-4
            or norm_error > 1e-5 or causal_error > 3e-5
            or row_error > 3e-5):
        raise ValueError(f"Stage155 compact parity/causality failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists() or GRAPH.exists():
        parser.error("Refusing to overwrite an existing qualification or graph")
    if sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected fixed baseline checkpoint")
    source_sha = sha(args.source)
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_rdrop"
            or int(payload["config"]["width"]) != 320
            or int(payload["config"]["depth"]) != 10
            or payload["config"]["conv_layers"] != [2, 4, 6, 8, 10]
            or int(payload.get("train_tokens", 0)) != 7200 * 32 * 256
            or payload.get("seed") != 17):
        raise ValueError("Expected the fixed Stage155 full-run checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    source, _ = make_model(payload["implementation"], payload["config"], device)
    source.load_state_dict(payload["model"], strict=True)
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_hybrid_conv_structured", config, device)
    deployed.load_state_dict(inference_state(payload["model"]), strict=True)
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)

    portable = FeatureWrapper(copy.deepcopy(deployed)).eval()
    eager = FeatureWrapper(deployed).eval()
    replaced = make_portable(portable)
    with torch.inference_mode():
        norm_error = float((portable(ids) - eager(ids)).abs().max())
    if replaced == 0 or norm_error > 3e-4:
        raise ValueError("Stage155 portable norm rewrite failed")
    args.run_dir.mkdir(parents=True)
    GRAPH.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            portable, ids, str(GRAPH), export_params=True, opset_version=17,
            do_constant_folding=True, input_names=["ids"], output_names=["hidden"],
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    graph_sha = sha(GRAPH)
    core = ov.Core()
    compiled = core.compile_model(str(GRAPH), "CPU", {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
    })
    with torch.inference_mode():
        actual = torch.from_numpy(next(iter(compiled({"ids": ids.numpy()}).values())))
        hidden_error = float((actual - eager(ids)).abs().max())
    if hidden_error > 3e-4:
        raise ValueError(f"OpenVINO trained hidden-state error {hidden_error}")

    compact_config = dict(config, feature_graph_sha256=graph_sha)
    candidate, candidate_module_sha = make_model(IMPLEMENTATION, compact_config, device)
    source_state = deployed.state_dict()
    keys = set(candidate.state_dict())
    if not keys <= set(source_state) or not any(
            name.startswith("blocks.") for name in set(source_state) - keys):
        raise ValueError("Compact head selection has unexpected keys")
    candidate.load_state_dict({key: source_state[key] for key in keys}, strict=True)
    candidate.eval()
    smoke_result = smoke(deployed, candidate)
    checkpoint = args.run_dir / "stage155-openvino-neural.pt"
    exported = dict(payload, implementation=IMPLEMENTATION,
                    config=compact_config, model=candidate.state_dict())
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=source_sha,
        graph_sha256=graph_sha,
        method="single_copy_fp32_openvino_features_neural_only_head",
        discarded_pytorch_feature_state_keys=len(set(source_state) - keys),
        no_new_gradient_targets=True, no_test_scoring=True,
        smoke=smoke_result)
    atomic_torch_save(exported, checkpoint)

    validation_path = args.run_dir / "validation-cpu-fp32.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation["targets"] != 376599 or validation["utf8_bytes"] != 1148007:
        raise ValueError("Complete validation coverage changed")
    resource_path = args.run_dir / "resources.json"
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
        protocol=PROTOCOL, status="stage155_compact_neural_resource_audited",
        source_checkpoint_sha256=source_sha, checkpoint_sha256=sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        implementation_sha256=candidate_module_sha,
        graph_sha256=graph_sha, graph_bytes=GRAPH.stat().st_size,
        max_portable_norm_error=norm_error,
        max_openvino_hidden_error=hidden_error,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        baseline_median_seconds=resources["baseline"]["median_seconds"],
        candidate_median_seconds=resources["candidate"]["median_seconds"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets, asset_sizes=asset_sizes,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        beats_stage143_validation=validation["bpb"] < STAGE143_BPB,
        smoke=smoke_result, inference_files=list(INFERENCE_FILES),
        source_hashes={name: sha(ROOT / name) for name in INFERENCE_FILES},
        split="validation", precision="fp32", no_test_scoring=True)
    final["qualified"] = all((final["beats_stage143_validation"],
                              final["within_five_x_time_limit"],
                              final["within_four_gib_peak_rss_limit"],
                              final["within_64mib_asset_limit"]))
    atomic_json_dump(final, args.run_dir / "final.json")
    if sha(args.source) != source_sha:
        raise ValueError("Source checkpoint changed during qualification")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
