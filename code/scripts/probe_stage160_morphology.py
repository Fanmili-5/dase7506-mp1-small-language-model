"""CPU and invariance preflight for the zero-start morphology output head."""
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
from torch.nn import functional as F

from common import PROTOCOL, sha
from student_stage160_morphology import MorphologyResidual, TOKENIZER_SHA256

STAGE143_ASSETS = 55_810_412
ARTIFACT_RESERVE = 200_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new preflight output")
    torch.manual_seed(160017)
    torch.set_num_threads(4)
    residual = MorphologyResidual().eval()
    base = F.log_softmax(torch.randn(32, 256, 2048), dim=-1)
    hidden = torch.randn(32, 256, 288)
    with torch.inference_mode():
        actual = residual(base, hidden)
        zero_error = float((actual - base).abs().max())
        normalization_error = float(actual.logsumexp(-1).abs().max())
        changed_hidden = hidden.clone()
        changed_hidden[:, 128:] += 1.0
        changed_base = base.clone()
        changed_base[:, 128:] = F.log_softmax(
            torch.randn_like(changed_base[:, 128:]), dim=-1)
        causal_error = float((actual[:, :128] - residual(
            changed_base, changed_hidden)[:, :128]).abs().max())
        row_error = float((actual[0] - residual(base[:1], hidden[:1])[0]).abs().max())
        for _ in range(2):
            residual(base, hidden)
        times = []
        for _ in range(8):
            before = time.perf_counter()
            residual(base, hidden)
            times.append(time.perf_counter() - before)
    median = statistics.median(times)
    parameters = sum(p.numel() for p in residual.parameters())
    parameter_bytes = sum(p.numel() * p.element_size()
                          for p in residual.parameters())
    assets = STAGE143_ASSETS + parameter_bytes + ARTIFACT_RESERVE
    result = dict(
        protocol=PROTOCOL, purpose="synthetic_input_head_only_preflight",
        split="none", no_test_scoring=True,
        source_sha256=sha(Path(__file__)),
        module_sha256=sha(ROOT / "student_stage160_morphology.py"),
        tokenizer_sha256=TOKENIZER_SHA256,
        residual_parameters=parameters, residual_parameter_bytes=parameter_bytes,
        artifact_reserve_bytes=ARTIFACT_RESERVE,
        conservative_projected_combined_assets_bytes=assets,
        max_zero_start_logp_error=zero_error,
        max_log_normalization_error=normalization_error,
        max_causal_prefix_error=causal_error,
        max_independent_row_error=row_error,
        head_only_timing_seconds=times,
        head_only_median_seconds=median,
        pass_zero_start=zero_error <= 2e-6,
        pass_normalization=normalization_error <= 1e-5,
        pass_causal_rows=causal_error <= 1e-6 and row_error <= 1e-6,
        pass_head_only_cpu_budget=median <= .45,
        pass_projected_assets=assets <= 64 * 1024 ** 2,
        warning="Head-only synthetic preflight; no complete CPU/RAM qualification",
    )
    result["training_pilot_authorized"] = all(result[key] for key in (
        "pass_zero_start", "pass_normalization", "pass_causal_rows",
        "pass_head_only_cpu_budget", "pass_projected_assets"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
