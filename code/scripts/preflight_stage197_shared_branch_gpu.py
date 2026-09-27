"""Synthetic-only GPU backward/causality preflight for Stage197."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup, sha
from train_experiment import training_autocast

CONFIG = ROOT / "configs/stage153_shared_branch_rdrop.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    torch.manual_seed(197017)
    torch.cuda.manual_seed_all(197017)
    model, implementation_sha = make_model("student_stage153_shared_branch", config, device)
    model.train()
    ids = torch.randint(0, 2048, (32, 256), device=device)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future_targets = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with training_autocast(device, precision):
        loss, parts = model.rdrop_training_loss(ids, targets, future_targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite one-update synthetic loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    model.eval()
    with torch.no_grad():
        small = ids[:2]
        logp = model.predict_log_probs(small)
        normalization = float(logp.logsumexp(-1).abs().max())
        changed = small.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 2048
        prefix_error = float((model.predict_log_probs(changed)[0, :-1]
                              - logp[0, :-1]).abs().max())
    result = {
        "status": "synthetic_gpu_preflight_only", "protocol": PROTOCOL,
        "implementation": "student_stage153_shared_branch",
        "implementation_sha256": implementation_sha,
        "config_sha256": sha(CONFIG),
        "preflight_source_sha256": sha(Path(__file__)),
        "device_name": torch.cuda.get_device_name(device),
        "precision": precision, "batch": 32, "context": 256,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic_loss": float(loss.detach()), "grad_norm": grad_norm,
        "max_normalization_error": normalization,
        "max_future_prefix_error": prefix_error,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "training_gate_passed": (
            torch.isfinite(loss).item() and grad_norm > 0
            and normalization <= 1e-3 and prefix_error <= 1e-5
            and reserved < torch.cuda.get_device_properties(device).total_memory
        ),
        "no_data_split_opened": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
