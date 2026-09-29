"""Exact-output and one-batch CPU timing screen for cached row maxima."""
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

from common import load_data, make_model, setup, sha, windows


STAGE115_SHA = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"


def install_derived_buffers(model):
    for table in model.ngram.tables:
        sizes = table.offsets[1:] - table.offsets[:-1]
        edge_rows = torch.repeat_interleave(torch.arange(table.keys.numel()), sizes)
        table.row_max.scatter_reduce_(0, edge_rows, table.mass,
                                      reduce="amax", include_self=True)
    with torch.no_grad():
        model.gate_alpha.copy_(model.gate_coeff / model.gate_std)
        model.gate_intercept.copy_(-(model.gate_mean * model.gate_alpha).sum())


def timed(model, ids, repeats=10):
    with torch.inference_mode():
        for _ in range(2):
            model.predict_log_probs(ids)
        samples = []
        for _ in range(repeats):
            before = time.perf_counter()
            model.predict_log_probs(ids)
            samples.append(time.perf_counter() - before)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage115", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage116 output file")
    if sha(args.stage115) != STAGE115_SHA:
        raise ValueError("Unexpected Stage115 checkpoint")
    payload = torch.load(args.stage115, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    candidate, _ = make_model("student_stage116_rowmax_gate", payload["config"], device)
    reference.load_state_dict(payload["model"], strict=True)
    mismatch = candidate.load_state_dict(payload["model"], strict=False)
    expected_missing = {f"ngram.tables.{i}.row_max" for i in range(4)}
    expected_missing |= {"gate_alpha", "gate_intercept"}
    if set(mismatch.missing_keys) != expected_missing or mismatch.unexpected_keys:
        raise ValueError("Derived-buffer state mismatch")
    install_derived_buffers(candidate)
    reference.eval(); candidate.eval()
    ids, _ = next(windows(load_data()["validation"][0], 32))
    with torch.inference_mode():
        original = reference.predict_log_probs(ids)
        altered = candidate.predict_log_probs(ids)
    max_probability_error = float((original.exp() - altered.exp()).abs().max())
    max_logp_error = float((original - altered).abs().max())
    if max_probability_error > 3e-6 or max_logp_error > 3e-5:
        raise ValueError("Cached row maximum changed probabilities")
    original_seconds = timed(reference, ids)
    candidate_seconds = timed(candidate, ids)
    result = dict(
        stage115_sha256=STAGE115_SHA, source_sha256=sha(Path(__file__)),
        candidate_implementation_sha256=sha(ROOT / "student_stage116_rowmax_gate.py"),
        max_probability_error=max_probability_error, max_logp_error=max_logp_error,
        original_seconds=original_seconds, candidate_seconds=candidate_seconds,
        original_median=statistics.median(original_seconds),
        candidate_median=statistics.median(candidate_seconds),
        relative_time=statistics.median(candidate_seconds)
        / statistics.median(original_seconds), no_test_scoring=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
