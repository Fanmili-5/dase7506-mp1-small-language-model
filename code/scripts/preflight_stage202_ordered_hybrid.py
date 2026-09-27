"""Synthetic-only admission check for the fixed local-first hybrid layout."""
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

CONFIG = ROOT / "configs/stage202_bottom_local_top_global.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
IMPLEMENTATION = "student_hybrid_conv_rdrop"


def check_config() -> tuple[dict, dict]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"conv_layers"} or config["conv_layers"] != [1, 2, 3, 4]:
        raise ValueError(f"Stage202 must change only fixed block order: {changed}")
    return config, control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    config, control = check_config()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(202017)
    torch.cuda.manual_seed_all(202017)
    model, implementation_sha = make_model(IMPLEMENTATION, config, device)
    control_model, _ = make_model(IMPLEMENTATION, control, torch.device("cpu"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    equal_parameters = parameters == sum(
        parameter.numel() for parameter in control_model.parameters())
    del control_model
    model.train()
    ids = torch.randint(0, 2048, (32, 256), device=device)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with training_autocast(device, precision):
        loss, _ = model.rdrop_training_loss(ids, targets, future)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite synthetic loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not 0 < grad_norm < float("inf"):
        raise FloatingPointError("Non-finite or zero synthetic gradient")
    optimizer.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    model.eval()
    with torch.inference_mode():
        small = ids[:2].clone()
        logp = model.predict_log_probs(small)
        normalization = float(logp.logsumexp(-1).abs().max())
        changed = small.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 2048
        future_prefix = float((model.predict_log_probs(changed)[0, :-1]
                               - logp[0, :-1]).abs().max())
        changed = small.clone()
        changed[1] = (changed[1] + 1) % 2048
        row_independence = float((model.predict_log_probs(changed)[0]
                                  - logp[0]).abs().max())
    total = torch.cuda.get_device_properties(device).total_memory
    passed = (equal_parameters and normalization <= 1e-5
              and future_prefix <= 1e-5 and row_independence <= 1e-5
              and allocated <= 7 * 1024**3 and reserved < total)
    result = dict(
        status="synthetic_gpu_preflight_only", protocol=PROTOCOL,
        implementation=IMPLEMENTATION, implementation_sha256=implementation_sha,
        config_sha256=sha(CONFIG), control_config_sha256=sha(CONTROL),
        source_sha256=sha(Path(__file__)), device_name=torch.cuda.get_device_name(device),
        precision=precision, batch=32, context=256, parameters=parameters,
        equal_parameter_count=equal_parameters, synthetic_optimizer_steps_completed=1,
        synthetic_loss=float(loss.detach()), grad_norm=grad_norm,
        max_normalization_error=normalization,
        max_future_prefix_error=future_prefix,
        max_row_independence_error=row_independence,
        peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
        gpu_total_bytes=total, training_gate_passed=passed,
        no_data_split_opened=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage202 synthetic admission gate failed")


if __name__ == "__main__":
    main()
