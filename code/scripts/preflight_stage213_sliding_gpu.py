"""Input-only batch-32 BF16 GPU gate for the unchanged Stage212 architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, sha
from scripts.preflight_stage212_sliding_local import matched_models
from train_experiment import training_autocast

CONFIG = ROOT / "configs/stage212_sliding_local_attention_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
STAGE212_PREFLIGHT = ROOT / "runs/stage212-sliding-preflight-b/result.json"
BASELINE_SECONDS = 23.786125500046182
STAGE143_SECONDS = 86.05110589996912


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage213 GPU screen")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control_config = json.loads(CONTROL.read_text(encoding="utf-8"))
    prior = json.loads(STAGE212_PREFLIGHT.read_text(encoding="utf-8-sig"))
    if (prior["config_sha256"] != sha(CONFIG)
            or prior["implementation_sha256"]
            != sha(ROOT / "student_stage212_sliding_local_rdrop.py")
            or prior["max_openvino_feature_error"] > 3e-4
            or prior["projected_assets_bytes"] > 64 * 1024**2
            or prior["shared_initial_state_tensors"] != 57
            or not prior["no_data_split_opened"]):
        raise ValueError("Stage212 fixed architecture or parity evidence changed")
    medians = prior["median_seconds"]
    projected_seconds = (STAGE143_SECONDS
                         + 46 * (medians["candidate"] - medians["reference"]))
    projected_ratio = projected_seconds / BASELINE_SECONDS
    if projected_ratio >= 5:
        raise ValueError("Even the course-style CPU projection exceeds 5x")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage213 GPU screen requires BF16 CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    control, model, implementation_sha, shared = matched_models(
        device, config, control_config)
    del control
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    ids = torch.randint(0, 2048, (32, 256), device=device)
    labels = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        with training_autocast(device, "bf16"):
            loss, _ = model.rdrop_training_loss(ids, labels, future)
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
        grad_norm = local_grad_norm = None
        error = "CUDA out of memory at physical batch 32"
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(error is None and 0 < grad_norm < float("inf")
                  and 0 < local_grad_norm < float("inf")
                  and allocated <= 7 * 1024**3 and reserved < total)
    result = dict(status="synthetic_gpu_preflight_only", protocol=PROTOCOL,
                  no_data_split_opened=True, test_scored=False,
                  config_sha256=sha(CONFIG), control_config_sha256=sha(CONTROL),
                  source_sha256=sha(Path(__file__)),
                  implementation_sha256=implementation_sha,
                  stage212_result_sha256=sha(STAGE212_PREFLIGHT),
                  shared_initial_state_tensors=shared,
                  projected_full_cpu_seconds=projected_seconds,
                  projected_full_cpu_ratio=projected_ratio,
                  projected_is_not_formal_resource_pass=True,
                  synthetic_loss=float(loss.detach()) if loss is not None else None,
                  synthetic_grad_norm=grad_norm,
                  local_qkv_grad_norm=local_grad_norm,
                  peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                  gpu_total_bytes=total, error=error,
                  admit_matched_quality_pilot=passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage213 GPU preflight gate failed")


if __name__ == "__main__":
    main()
