"""Input-only CPU/GPU feasibility screen for shared-depth Stage221."""
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

from common import PROTOCOL, sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper, make_portable, ov_model
from train_experiment import training_autocast
import student_stage221_tied_depth_rdrop as training
import student_stage221_tied_depth_structured as inference


REFERENCE = ROOT / "inference_assets/stage143-stage92-features.onnx"
REFERENCE_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
CONFIG = ROOT / "configs/stage221_tied_depth_rdrop.json"
CONTROL = ROOT / "configs/stage150_depth10_hybrid_rdrop.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new Stage221 preflight directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Stage143 reference graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"shared_tail_pair"} or config["shared_tail_pair"] != [7, 8]:
        raise ValueError("Stage221 changed outside the fixed shared tail pair")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = inference.build_model(training.inference_config(config)).eval()
    if model.blocks[8] is not model.blocks[6] or model.blocks[9] is not model.blocks[7]:
        raise ValueError("Tail sharing was not installed")
    ids = torch.randint(0, 2048, (32, 256), generator=torch.Generator().manual_seed(221017))
    with torch.inference_mode():
        logp = model.predict_log_probs(ids[:2, :64])
        norm_error = float(logp.logsumexp(-1).abs().max())
        changed_ids = ids[:2, :64].clone()
        changed_ids[:, 32:] = (changed_ids[:, 32:] + 97) % 2048
        causal_error = float((logp[:, :32]
                              - model.predict_log_probs(changed_ids)[:, :32]).abs().max())
        row_error = float((logp[:1]
                           - model.predict_log_probs(ids[:1, :64])).abs().max())
    eager = FeatureWrapper(model).eval()
    portable = FeatureWrapper(copy.deepcopy(model)).eval()
    rewritten = make_portable(portable)
    with torch.inference_mode():
        portable_error = float((portable(ids[:1]) - eager(ids[:1])).abs().max())
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage221-random-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(portable, ids, str(graph), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["ids"], output_names=["hidden"],
                          dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    candidate = ov_model(core, graph)
    ov_errors = []
    with torch.inference_mode():
        for batch in (ids, ids[:1]):
            actual = torch.from_numpy(next(iter(candidate({"ids": batch.numpy()}).values())))
            ov_errors.append(float((actual - eager(batch)).abs().max()))
    times = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()})
        candidate({"ids": ids.numpy()})
    for index in range(8):
        order = (("reference", reference), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for label, compiled in order:
            started = time.perf_counter()
            compiled({"ids": ids.numpy()})
            times[label].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in times.items()}
    feature_ratio = medians["candidate"] / medians["reference"]
    projected_assets = graph.stat().st_size + 25_000_000 + 200_000
    unique_parameters = sum(p.numel() for p in model.parameters())
    del model, eager, portable

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage221 requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    gpu = training.build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    losses, grad_norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, "bf16"):
            loss, _ = gpu.rdrop_training_loss(ids.to(device), targets, future)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage221 loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(gpu.parameters(), 1.0))
        if not 0 < norm < float("inf"):
            raise FloatingPointError("Nonfinite Stage221 gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        grad_norms.append(norm)
    torch.cuda.synchronize(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(norm_error <= 1e-5 and causal_error <= 3e-5
                  and row_error <= 1e-5 and rewritten > 0
                  and portable_error <= 3e-4 and max(ov_errors) <= 3e-4
                  and feature_ratio <= 1.25
                  and projected_assets <= 64 * 1024**2
                  and reserved < total and reserved < 8 * 1024**3)
    result = dict(
        status="synthetic_input_only_preflight", protocol=PROTOCOL,
        no_validation_or_test_scoring=True,
        config_sha256=sha(CONFIG), control_config_sha256=sha(CONTROL),
        source_sha256=sha(Path(__file__)),
        training_model_sha256=sha(ROOT / "student_stage221_tied_depth_rdrop.py"),
        inference_model_sha256=sha(ROOT / "student_stage221_tied_depth_structured.py"),
        core_model_sha256=sha(ROOT / "student_stage221_tied_depth.py"),
        reference_graph_sha256=REFERENCE_SHA,
        candidate_graph_sha256=sha(graph), candidate_graph_bytes=graph.stat().st_size,
        projected_conservative_assets_bytes=projected_assets,
        unique_neural_parameters=unique_parameters,
        max_normalization_error=norm_error,
        max_future_prefix_error=causal_error,
        max_independent_row_error=row_error,
        portable_norm_count=rewritten,
        portable_hidden_max_error=portable_error,
        openvino_hidden_max_error=max(ov_errors),
        timing_seconds=times, median_seconds=medians,
        candidate_to_reference_feature_time=feature_ratio,
        synthetic_losses=losses, synthetic_grad_norms=grad_norms,
        synthetic_optimizer_steps=2,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        gpu_peak_reserved_bytes=reserved, gpu_total_bytes=total,
        admit_matched_training_pilot=passed,
    )
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage221 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
