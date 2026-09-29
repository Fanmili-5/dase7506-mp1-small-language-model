"""Cache exact Stage143 validation-target log probabilities for a diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows

EXPECTED_CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
EXPECTED_BPB = 1.399686162042141


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        p.error("Use a new diagnostic output path")
    if sha(args.checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("Unexpected Stage143 checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_stage143_openvino_singlepass":
        raise ValueError("Unexpected checkpoint protocol/implementation")
    model, source_sha = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"])
    model.eval()
    validation, byte_count = load_data()["validation"]
    chunks = []
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            logp = model.predict_log_probs(x).float()
            target = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            chunks.append(target[y != -100].double().cpu().numpy())
    values = np.concatenate(chunks)
    if len(values) != len(validation) - 1 or not np.isfinite(values).all():
        raise ValueError("Incomplete or invalid target log probabilities")
    bpb = -float(values.sum()) / np.log(2) / byte_count
    if abs(bpb - EXPECTED_BPB) > 2e-5:
        raise ValueError(f"Stage143 BPB mismatch: {bpb}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, values)
    result = {
        "purpose": "validation_target_probability_diagnostic_only",
        "protocol": PROTOCOL,
        "split": "validation",
        "test_scored": False,
        "targets": len(values),
        "utf8_bytes": byte_count,
        "bpb": bpb,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "implementation_sha256": source_sha,
        "source_sha256": sha(Path(__file__)),
        "array_sha256": sha(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
