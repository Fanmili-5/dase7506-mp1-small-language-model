"""Synthetic-only Stage207 lexical feature, output and resource screen."""
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
import student_stage207_prefix_residual as model_source


STAGE143_ASSETS = 55_810_412
ASSET_LIMIT = 64 * 1024**2
SOURCE_RESERVE = 524_288


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage207 preflight evidence")
    torch.manual_seed(207017)
    torch.set_num_threads(4)
    start = next(i for i, value in enumerate(model_source.STARTS)
                 if value is not None and len(value) == 2)
    cont = next(i for i, value in enumerate(model_source.CONTINUATIONS)
                if value is not None and len(value) == 1)
    ids = torch.tensor([[start, cont] * 128] * 32, dtype=torch.long)
    base = F.log_softmax(torch.randn(32, 256, 2048), dim=-1)
    hidden = torch.randn(32, 256, 288)
    residual = model_source.PrefixConditionedResidual().eval()
    with torch.inference_mode():
        actual = residual(base, hidden, ids)
        zero_error = float((actual - base).abs().max())
        norm_error = float(actual.logsumexp(-1).abs().max())
        short_ids = ids[:2, :12]
        short_base = base[:2, :12]
        short_hidden = hidden[:2, :12]
        original = residual(short_base, short_hidden, short_ids)
        changed_ids = short_ids.clone()
        changed_ids[0, -1] = start
        future_error = float((residual(short_base, short_hidden, changed_ids)[0, :-1]
                              - original[0, :-1]).abs().max())
        row_error = float((residual(short_base[:1], short_hidden[:1], short_ids[:1])[0]
                           - original[0]).abs().max())
        for _ in range(2):
            residual(base, hidden, ids)
        timings = []
        for _ in range(8):
            started = time.perf_counter()
            residual(base, hidden, ids)
            timings.append(time.perf_counter() - started)
    parameters = sum(p.numel() for p in residual.parameters())
    projected_assets = STAGE143_ASSETS + 4 * parameters + SOURCE_RESERVE
    median = statistics.median(timings)
    passed = bool(zero_error <= 2e-6 and norm_error <= 1e-5
                  and future_error <= 1e-6 and row_error <= 1e-6
                  and projected_assets <= ASSET_LIMIT and median <= .25)
    result = {
        "status": "synthetic_input_only_preflight", "protocol": PROTOCOL,
        "no_data_split_opened": True, "test_scored": False,
        "source_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "student_stage207_prefix_residual.py"),
        "parent_source_sha256": sha(ROOT / "student_stage160_morphology.py"),
        "tokenizer_sha256": model_source.TOKENIZER_SHA256,
        "reference_stage143_assets_bytes": STAGE143_ASSETS,
        "residual_parameters": parameters,
        "projected_conservative_assets_bytes": projected_assets,
        "asset_limit_bytes": ASSET_LIMIT,
        "zero_start_max_logp_error": zero_error,
        "max_normalization_error": norm_error,
        "max_future_prefix_error": future_error,
        "max_independent_row_error": row_error,
        "batch": 32, "context": 256, "threads": 4,
        "head_feature_seconds": timings,
        "median_head_feature_seconds": median,
        "admit_train_only_pilot": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage207 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
