"""Train-only basis, input contract and synthetic CPU cost screen for Stage215."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import sha
from scripts.build_stage215_semantic_basis import build_basis
from student_hybrid_conv_rdrop import build_model as control_model
from student_stage215_semantic_input import build_model as candidate_model


CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
CANDIDATE = ROOT / "configs/stage215_semantic_input_rdrop.json"
STAGE143_ASSET_BYTES = 55_810_412
ASSET_RESERVE_BYTES = 200_000


def timed(model: torch.nn.Module, ids: torch.Tensor) -> float:
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.predict_log_probs(ids)
    if output.shape != (*ids.shape, 2048) or not torch.isfinite(output).all():
        raise ValueError("Invalid synthetic output")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new preflight output")
    torch.set_num_threads(4)
    control_config = json.loads(CONTROL.read_text(encoding="utf-8"))
    config = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    changed = {key for key in control_config.keys() | config.keys()
               if control_config.get(key) != config.get(key)}
    if changed != {"semantic_dim"} or config["semantic_dim"] != 64:
        raise ValueError("Stage215 config changed outside the fixed input mechanism")
    torch.manual_seed(17)
    control = control_model(control_config).eval()
    torch.manual_seed(17)
    candidate = candidate_model(config).eval()
    for name, expected in control.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], expected,
                                   rtol=0, atol=0)
    basis, basis_meta = build_basis()
    candidate.set_semantic_basis(basis)
    generator = torch.Generator().manual_seed(215017)
    ids = torch.randint(2048, (32, 256), generator=generator)
    with torch.inference_mode():
        reference = control.predict_log_probs(ids)
        zero_start = candidate.predict_log_probs(ids)
    parity = float((reference - zero_start).abs().max())
    normalization = float(zero_start.logsumexp(-1).abs().max())
    if parity > 1e-6 or normalization > 1e-5:
        raise ValueError("Zero-start contract failed")
    for _ in range(2):
        timed(control, ids)
        timed(candidate, ids)
    control_times, candidate_times = [], []
    for _ in range(4):
        control_times.append(timed(control, ids))
        candidate_times.append(timed(candidate, ids))
    control_median = statistics.median(control_times)
    candidate_median = statistics.median(candidate_times)
    ratio = candidate_median / control_median
    projected_assets = (STAGE143_ASSET_BYTES + basis_meta["basis_bytes"]
                        + candidate.semantic_projection.weight.numel() * 4
                        + ASSET_RESERVE_BYTES)
    result = {
        "purpose": "input_only_stage215_cpu_preflight_not_full_resource_qualification",
        "no_validation_or_test_scoring": True,
        "basis": basis_meta,
        "config_sha256": sha(CANDIDATE),
        "control_config_sha256": sha(CONTROL),
        "implementation_sha256": sha(ROOT / "student_stage215_semantic_input.py"),
        "source_sha256": sha(Path(__file__)),
        "shared_initial_tensors_exact": True,
        "zero_start_max_logp_error": parity,
        "normalization_max_error": normalization,
        "control_seconds": control_times,
        "candidate_seconds": candidate_times,
        "control_median_seconds": control_median,
        "candidate_median_seconds": candidate_median,
        "candidate_to_control_time_ratio": ratio,
        "projected_inference_asset_bytes": projected_assets,
        "timing_gate_pass": ratio <= 1.20,
        "asset_projection_gate_pass": projected_assets <= 64 * 1024**2,
        "admit_quality_pilot": ratio <= 1.20 and projected_assets <= 64 * 1024**2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
