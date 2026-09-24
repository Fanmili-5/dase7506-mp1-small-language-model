"""Label-using validation oracle for one heterogeneous expert per window."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage90_heterogeneous_ensemble import (
    ALTERNATE_SHA, PRIMARY_SHA,
)
from scripts.train_stage86_calibration_aware import (
    calibrated_neural_log_probs, train_log_prior,
)
from train_experiment import atomic_json_dump


EXPECTED_TARGETS = 376599
EXPECTED_BYTES = 1148007
EXPECTED_PRIMARY_BPB = 1.4091613369059282


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage138 evidence")
    if sha(args.primary) != PRIMARY_SHA or sha(args.alternate) != ALTERNATE_SHA:
        raise ValueError("Unexpected frozen expert checkpoint")
    primary_payload = torch.load(args.primary, map_location="cpu", weights_only=True)
    alternate_payload = torch.load(args.alternate, map_location="cpu", weights_only=True)
    if (primary_payload.get("protocol") != PROTOCOL
            or alternate_payload.get("protocol") != PROTOCOL
            or primary_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or alternate_payload.get("implementation")
            != "student_hybrid_conv_structured"):
        raise ValueError("Unexpected expert payload")
    device, _ = setup("cuda", "fp32", 4)
    primary, _ = make_model(primary_payload["implementation"],
                            primary_payload["config"], device)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    primary.load_state_dict(primary_payload["model"], strict=True)
    alternate.load_state_dict(alternate_payload["model"], strict=True)
    primary.eval(); alternate.eval()
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    validation, raw_bytes = load_data()["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Validation byte count changed")
    primary_losses, alternate_losses = [], []
    targets = 0
    with torch.inference_mode():
        for ids_cpu, labels_cpu in windows(validation, 32):
            ids = ids_cpu.to(device)
            labels = labels_cpu.to(device)
            valid = labels != -100
            gather = labels.clamp_min(0).unsqueeze(-1)
            primary_logp = calibrated_neural_log_probs(
                primary, ids, log_prior).gather(-1, gather).squeeze(-1)
            alternate_logp = alternate.predict_log_probs(ids).gather(
                -1, gather).squeeze(-1)
            primary_losses.append((-primary_logp.double() * valid).sum(-1).cpu())
            alternate_losses.append((-alternate_logp.double() * valid).sum(-1).cpu())
            targets += int(valid.sum())
    if targets != EXPECTED_TARGETS:
        raise ValueError("Validation target coverage changed")
    p = torch.cat(primary_losses)
    a = torch.cat(alternate_losses)
    if p.shape != a.shape or p.numel() != math.ceil(EXPECTED_TARGETS / 256):
        raise ValueError("Independent window alignment changed")
    denominator = math.log(2) * raw_bytes
    primary_bpb = float(p.sum()) / denominator
    if abs(primary_bpb - EXPECTED_PRIMARY_BPB) > 2e-5:
        raise ValueError(f"Stage90 primary control mismatch: {primary_bpb}")
    choose_alternate = a < p
    oracle_nll = float(torch.minimum(p, a).sum())
    advantages = (p - a).abs()
    result = dict(
        protocol=PROTOCOL, split="validation", precision="fp32",
        purpose="label_using_oracle_diagnostic_not_deployable",
        primary_sha256=PRIMARY_SHA, alternate_sha256=ALTERNATE_SHA,
        source_sha256=sha(Path(__file__)),
        primary_bpb=primary_bpb, alternate_bpb=float(a.sum()) / denominator,
        oracle_bpb=oracle_nll / denominator,
        oracle_nll_nats=oracle_nll,
        oracle_alternate_windows=int(choose_alternate.sum()),
        independent_windows=int(p.numel()),
        median_hindsight_window_advantage_nats=float(advantages.median()),
        mean_hindsight_window_advantage_nats=float(advantages.mean()),
        targets=targets, utf8_bytes=raw_bytes,
        train_unigram_tokens=train_tokens, no_test_scoring=True,
        no_deployable_router=True,
        passes_exploratory_ceiling_gate=oracle_nll / denominator < 1.38,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
