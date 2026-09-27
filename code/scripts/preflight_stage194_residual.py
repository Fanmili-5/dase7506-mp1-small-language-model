"""Synthetic-only CPU feasibility screen for the Stage194 residual architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup
from student_stage194_residual import ResidualGatedLM

BASE_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
BASE_ASSETS_BYTES = 55_810_412
LIMIT_BYTES = 67_108_864
SOURCE_RESERVE = 262_144


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(model: torch.nn.Module, ids: torch.Tensor) -> float:
    with torch.no_grad():
        start = time.perf_counter()
        result = model(ids)
        elapsed = time.perf_counter() - start
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite model output")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing preflight evidence")
    if sha(args.checkpoint) != BASE_SHA:
        raise ValueError("Stage143 checkpoint changed")
    device, precision = setup("cpu", "fp32", args.threads)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_stage143_openvino_singlepass":
        raise ValueError("Unexpected frozen base implementation or protocol")
    base, base_module_sha = make_model(payload["implementation"], payload["config"], device)
    base.load_state_dict(payload["model"], strict=True)
    base.eval()
    candidate = ResidualGatedLM(base).to(device).eval()
    torch.manual_seed(194)
    ids = torch.randint(0, 2048, (2, 256), device=device)
    with torch.no_grad():
        base_logp = base(ids)
        candidate_logp = candidate(ids)
        parity = float((base_logp - candidate_logp).abs().max())
        mass_error = float((candidate_logp.exp().sum(-1) - 1).abs().max())
        altered = ids.clone()
        altered[0, -1] = (altered[0, -1] + 1) % 2048
        causal = float((candidate(altered)[0, :-1] - candidate_logp[0, :-1]).abs().max())
    expert_bytes = sum(t.numel() * t.element_size() for t in candidate.expert.state_dict().values())
    projected_assets = BASE_ASSETS_BYTES + expert_bytes + SOURCE_RESERVE
    timing_ids = torch.randint(0, 2048, (8, 256), device=device)
    base_times, candidate_times = [], []
    for _ in range(2):
        timed(base, timing_ids)
        timed(candidate, timing_ids)
    for index in range(8):
        if index % 2:
            candidate_times.append(timed(candidate, timing_ids))
            base_times.append(timed(base, timing_ids))
        else:
            base_times.append(timed(base, timing_ids))
            candidate_times.append(timed(candidate, timing_ids))
    ratio = statistics.median(candidate_times) / statistics.median(base_times)
    result = {
        "status": "synthetic_cpu_preflight_only", "protocol": PROTOCOL,
        "checkpoint_sha256": BASE_SHA, "base_implementation_sha256": base_module_sha,
        "residual_source_sha256": sha(ROOT / "student_stage194_residual.py"),
        "preflight_source_sha256": sha(Path(__file__)),
        "device": str(device), "precision": precision, "threads": args.threads,
        "max_parity_logp": parity, "max_prefix_causality_logp": causal,
        "max_probability_mass_error": mass_error,
        "expert_state_bytes": expert_bytes, "projected_assets_bytes": projected_assets,
        "asset_limit_bytes": LIMIT_BYTES,
        "median_base_seconds": statistics.median(base_times),
        "median_candidate_seconds": statistics.median(candidate_times),
        "candidate_base_cpu_ratio": ratio,
        "feasibility_gate_passed": (
            parity <= 1e-5 and causal <= 1e-5 and mass_error <= 1e-5
            and projected_assets <= LIMIT_BYTES and ratio <= 1.20
        ),
        "no_training_data_opened": True, "no_validation_data_opened": True,
        "no_test_data_opened": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
