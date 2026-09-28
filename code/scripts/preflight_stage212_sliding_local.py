"""Synthetic-only Stage212 exact-causality, OpenVINO and GPU feasibility screen."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch

from common import PROTOCOL, make_model, sha
from scripts.probe_stage145_attention_allocation import (
    FeatureWrapper, make_portable, ov_model, REFERENCE, REFERENCE_SHA,
)
from train_experiment import training_autocast
import student_stage212_sliding_local_rdrop as training_module
import student_stage212_sliding_local_structured as inference_module


CONFIG = ROOT / "configs/stage212_sliding_local_attention_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
IMPLEMENTATION = "student_stage212_sliding_local_rdrop"
BASE_INFERENCE_ASSET_BYTES = 55_810_412
BASE_FEATURE_GRAPH_BYTES = 31_805_041


def matched_models(device: torch.device, config: dict, control_config: dict):
    """Copy every identical Stage54 tensor, leaving only novel local tensors random."""
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    control, _ = make_model("student_hybrid_conv_rdrop", control_config, device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    candidate, digest = make_model(IMPLEMENTATION, config, device)
    source, target = control.state_dict(), candidate.state_dict()
    common = set(source) & set(target)
    for name in common:
        if source[name].shape != target[name].shape:
            raise ValueError("Shared tensor changed shape: " + name)
        target[name].copy_(source[name])
    for name in set(source) ^ set(target):
        if not name.startswith(("blocks.1.", "blocks.3.",
                                "blocks.5.", "blocks.7.")):
            raise ValueError("Unexpected nonlocal tensor change: " + name)
    candidate.load_state_dict(target, strict=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return control, candidate, digest, len(common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Refusing to overwrite Stage212 preflight evidence")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Stage143 reference graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control_config = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control_config.keys()
               if config.get(key) != control_config.get(key)}
    if (changed != {"conv_layers", "local_attention_layers", "local_attention_window"}
            or config["local_attention_layers"] != [2, 4, 6, 8]
            or config["local_attention_window"] != 7):
        raise ValueError("Stage212 changed outside the fixed local mixer")
    torch.set_num_threads(4)
    control, candidate, candidate_sha, common_count = matched_models(
        torch.device("cpu"), config, control_config)
    control_params = sum(p.numel() for p in control.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    candidate.eval()
    ids = torch.randint(0, 2048, (32, 256), dtype=torch.long)
    with torch.inference_mode():
        probe = candidate.predict_log_probs(ids[:2, :24])
        normalization = float(probe.logsumexp(-1).abs().max())
        changed_ids = ids[:2, :24].clone()
        changed_ids[0, -1] = (changed_ids[0, -1] + 1) % 2048
        prefix_error = float((candidate.predict_log_probs(changed_ids)[0, :-1]
                              - probe[0, :-1]).abs().max())
        row_error = float((candidate.predict_log_probs(ids[:1, :24])[0]
                           - probe[0]).abs().max())
    deployed = inference_module.build_model(
        training_module.inference_config(config)).eval()
    deployed.load_state_dict(training_module.inference_state(candidate.state_dict()),
                             strict=True)
    with torch.inference_mode():
        export_error = float((deployed.predict_log_probs(ids[:2, :24])
                              - probe).abs().max())
    eager = FeatureWrapper(deployed).eval()
    portable = FeatureWrapper(copy.deepcopy(deployed)).eval()
    replaced_norms = make_portable(portable)
    with torch.inference_mode():
        portable_error = float((portable(ids[:2]) - eager(ids[:2])).abs().max())
    if replaced_norms == 0 or portable_error > 3e-4:
        raise ValueError("Portable model changed the random-weight features")
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage212-random-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(portable, ids, str(graph), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["ids"], output_names=["hidden"],
                          dynamic_axes={"ids": {0: "batch"},
                                        "hidden": {0: "batch"}})
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    compiled = ov_model(core, graph)
    errors = []
    with torch.inference_mode():
        for batch in (ids, ids[:1]):
            expected = eager(batch)
            actual = torch.from_numpy(next(iter(compiled({"ids": batch.numpy()}).values())))
            errors.append(float((actual - expected).abs().max()))
    timings = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()}); compiled({"ids": ids.numpy()})
    for index in range(8):
        pairs = (("reference", reference), ("candidate", compiled))
        if index % 2:
            pairs = tuple(reversed(pairs))
        for label, model in pairs:
            started = time.perf_counter()
            model({"ids": ids.numpy()})
            timings[label].append(time.perf_counter() - started)
    medians = {label: statistics.median(values) for label, values in timings.items()}
    projected_assets = (BASE_INFERENCE_ASSET_BYTES - BASE_FEATURE_GRAPH_BYTES
                        + graph.stat().st_size
                        + max(0, candidate_params - control_params) * 4 + 300_000)
    cpu_pass = bool(max(errors) <= 3e-4
                    and max(normalization, prefix_error, row_error, export_error) <= 1e-5
                    and projected_assets <= 64 * 1024**2
                    and medians["candidate"] <= 1.20 * medians["reference"])
    gpu = {}
    if cpu_pass:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Stage212 GPU screen requires BF16 CUDA")
        device = torch.device("cuda")
        gpu_control, model, _, gpu_common = matched_models(
            device, config, control_config)
        del gpu_control
        if gpu_common != common_count:
            raise ValueError("CPU/GPU shared state count differs")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      betas=(.9, .999), weight_decay=.1)
        gpu_ids = torch.randint(0, 2048, (32, 256), device=device)
        labels = torch.randint(0, 2048, (32, 256), device=device)
        future = torch.randint(0, 2048, (2, 32, 256), device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            with training_autocast(device, "bf16"):
                loss, _ = model.rdrop_training_loss(gpu_ids, labels, future)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite synthetic loss")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            local_grad_norm = float(model.blocks[1].qkv.weight.grad.norm())
            optimizer.step()
            error = None
        except RuntimeError as caught:
            if "out of memory" not in str(caught).lower():
                raise
            loss = None
            grad_norm = None
            local_grad_norm = None
            error = "CUDA out of memory at physical batch 32"
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        gpu = dict(synthetic_loss=float(loss.detach()) if loss is not None else None,
                   synthetic_grad_norm=grad_norm,
                   local_qkv_grad_norm=local_grad_norm,
                   peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                   gpu_total_bytes=total, error=error)
        gpu_pass = bool(error is None and 0 < grad_norm < float("inf")
                        and 0 < local_grad_norm < float("inf")
                        and allocated <= 7 * 1024**3 and reserved < total)
    else:
        gpu_pass = False
    passed = cpu_pass and gpu_pass
    result = dict(status="synthetic_input_only_preflight", protocol=PROTOCOL,
                  no_data_split_opened=True, test_scored=False,
                  config_sha256=sha(CONFIG), control_config_sha256=sha(CONTROL),
                  source_sha256=sha(Path(__file__)),
                  implementation_sha256=candidate_sha,
                  shared_initial_state_tensors=common_count,
                  control_parameters=control_params, candidate_parameters=candidate_params,
                  max_normalization_error=normalization,
                  max_future_prefix_error=prefix_error,
                  max_independent_row_error=row_error,
                  max_inference_export_logp_error=export_error,
                  max_portable_feature_error=portable_error,
                  max_openvino_feature_error=max(errors),
                  graph_sha256=sha(graph), graph_bytes=graph.stat().st_size,
                  timings_seconds=timings, median_seconds=medians,
                  feature_cpu_ratio=medians["candidate"] / medians["reference"],
                  projected_assets_bytes=projected_assets,
                  cpu_asset_gate_passed=cpu_pass, gpu=gpu,
                  admit_matched_training_pilot=passed)
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage212 fixed preflight gate failed")


if __name__ == "__main__":
    main()
