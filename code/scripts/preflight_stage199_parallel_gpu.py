"""Synthetic-only batch-32 GPU feasibility check for the fixed Stage177 model."""
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

CONFIG = ROOT / "configs/stage177_parallel_mixer_rdrop.json"
IMPLEMENTATION = "student_stage177_parallel_mixer_rdrop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing preflight evidence")
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["width"] != 288 or config["depth"] != 8
            or config["conv_layers"] != [2, 4, 6, 8]):
        raise ValueError("Unexpected Stage177 configuration")
    torch.manual_seed(199017)
    torch.cuda.manual_seed_all(199017)
    model, implementation_sha = make_model(IMPLEMENTATION, config, device)
    model.train()
    ids = torch.randint(0, 2048, (32, 256), device=device)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    error = None
    loss_value = None
    grad_norm = None
    for _ in range(2):
        try:
            optimizer.zero_grad(set_to_none=True)
            with training_autocast(device, precision):
                loss, _ = model.rdrop_training_loss(ids, targets, future)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite synthetic loss")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not torch.isfinite(torch.tensor(grad_norm)):
                raise FloatingPointError("Non-finite synthetic gradient norm")
            optimizer.step()
            loss_value = float(loss.detach())
        except RuntimeError as caught:
            if "out of memory" not in str(caught).lower():
                raise
            error = "CUDA out of memory at fixed physical batch 32"
            break
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    normalization = None
    future_prefix = None
    if error is None:
        model.eval()
        with torch.inference_mode():
            small = ids[:2]
            logp = model.predict_log_probs(small)
            normalization = float(logp.logsumexp(-1).abs().max())
            changed = small.clone()
            changed[0, -1] = (changed[0, -1] + 1) % 2048
            future_prefix = float((model.predict_log_probs(changed)[0, :-1]
                                   - logp[0, :-1]).abs().max())
    total = torch.cuda.get_device_properties(device).total_memory
    passed = (error is None and grad_norm is not None and grad_norm > 0
              and normalization is not None and normalization <= 1e-3
              and future_prefix is not None and future_prefix <= 1e-5
              and reserved < total)
    result = {
        "status": "synthetic_gpu_preflight_only", "protocol": PROTOCOL,
        "implementation": IMPLEMENTATION,
        "implementation_sha256": implementation_sha,
        "config_sha256": sha(CONFIG),
        "source_sha256": sha(Path(__file__)),
        "device_name": torch.cuda.get_device_name(device),
        "precision": precision, "batch": 32, "context": 256,
        "synthetic_optimizer_steps_completed": 2 if error is None else 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic_loss": loss_value, "grad_norm": grad_norm,
        "max_normalization_error": normalization,
        "max_future_prefix_error": future_prefix,
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "gpu_total_bytes": total, "error": error,
        "training_gate_passed": passed,
        "no_data_split_opened": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
