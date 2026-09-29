"""Input-only Stage211 exact-start, inference strip and CUDA-memory screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, sha
from train_experiment import training_autocast
import student_hybrid_conv_rdrop


CONFIG = ROOT / "configs/stage211_long_horizon_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
OFFSETS = [2, 3, 8, 16, 32, 64]


def build_matched_candidate(config: dict, control_config: dict,
                            device: torch.device):
    """Match all shared seed-17 weights and the control post-init RNG state."""
    control, control_sha = make_model("student_hybrid_conv_rdrop", control_config, device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    candidate, candidate_sha = make_model("student_hybrid_conv_rdrop", config, device)
    source = control.state_dict()
    target = candidate.state_dict()
    for name, value in source.items():
        if name not in target or target[name].shape != value.shape:
            raise ValueError("Changed shared state: " + name)
        target[name].copy_(value)
    candidate.load_state_dict(target, strict=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if control_sha != candidate_sha:
        raise ValueError("Changed shared model implementation")
    return control, candidate, control_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage211 preflight evidence")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control_config = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control_config.keys()
               if config.get(key) != control_config.get(key)}
    if (changed != {"future_prediction_offsets"}
            or config["future_prediction_offsets"] != OFFSETS):
        raise ValueError("Stage211 changed outside the fixed auxiliary offsets")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    control, candidate, implementation_sha = build_matched_candidate(
        config, control_config, torch.device("cpu"))
    control.eval(); candidate.eval()
    ids = torch.randint(0, 2048, (2, 12), dtype=torch.long)
    with torch.inference_mode():
        control_logp = control.predict_log_probs(ids)
        candidate_logp = candidate.predict_log_probs(ids)
        zero_error = float((candidate_logp - control_logp).abs().max())
        normalization = float(candidate_logp.logsumexp(-1).abs().max())
        changed_ids = ids.clone()
        changed_ids[0, -1] = (changed_ids[0, -1] + 1) % 2048
        prefix_error = float((candidate.predict_log_probs(changed_ids)[0, :-1]
                              - candidate_logp[0, :-1]).abs().max())
        row_error = float((candidate.predict_log_probs(ids[:1])[0]
                           - candidate_logp[0]).abs().max())
    export_config_equal = (student_hybrid_conv_rdrop.inference_config(config)
                           == student_hybrid_conv_rdrop.inference_config(control_config))
    export_state = student_hybrid_conv_rdrop.inference_state(candidate.state_dict())
    control_export_state = student_hybrid_conv_rdrop.inference_state(control.state_dict())
    export_keys_equal = export_state.keys() == control_export_state.keys()
    export_error = max(float((export_state[key] - control_export_state[key]).abs().max())
                       for key in export_state) if export_keys_equal else float("inf")
    if any(name.startswith("future_") for name in export_state):
        raise ValueError("Future training heads leaked into inference state")
    del control, candidate

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage211 preflight requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    gpu_control, model, _ = build_matched_candidate(config, control_config, device)
    del gpu_control
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    gpu_ids = torch.randint(0, 2048, (32, 256), device=device)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (len(OFFSETS), 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with training_autocast(device, "bf16"):
        loss, _ = model.rdrop_training_loss(gpu_ids, targets, future)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Stage211 synthetic loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    long_grad_norm = float(model.future_projections[-1].weight.grad.norm())
    optimizer.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(zero_error == 0.0 and normalization <= 1e-5
                  and prefix_error <= 1e-5 and row_error <= 1e-5
                  and export_config_equal and export_keys_equal
                  and export_error == 0.0
                  and 0 < grad_norm < float("inf")
                  and 0 < long_grad_norm < float("inf")
                  and allocated <= 7 * 1024**3 and reserved < gpu_total)
    result = {
        "status": "synthetic_input_only_preflight", "protocol": PROTOCOL,
        "no_data_split_opened": True, "test_scored": False,
        "config_sha256": sha(CONFIG), "control_config_sha256": sha(CONTROL),
        "source_sha256": sha(Path(__file__)),
        "implementation_sha256": implementation_sha,
        "future_prediction_offsets": OFFSETS,
        "max_initial_inference_logp_error": zero_error,
        "max_normalization_error": normalization,
        "max_future_prefix_error": prefix_error,
        "max_independent_row_error": row_error,
        "export_config_equal": export_config_equal,
        "export_keys_equal": export_keys_equal,
        "max_export_state_error": export_error,
        "synthetic_loss": float(loss.detach()),
        "synthetic_grad_norm": grad_norm,
        "synthetic_long_head_grad_norm": long_grad_norm,
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
        "gpu_total_bytes": gpu_total,
        "admit_matched_training_pilot": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage211 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
