"""Score fixed Stage143/compact-expert mixture weights on validation only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows

STAGE143_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
STAGE143_BPB = 1.399686162042141
WEIGHTS = (0.0, 0.1, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage143-cache", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic output")
    meta = json.loads(args.stage143_cache.with_suffix(".json").read_text())
    if (sha(args.stage143_cache) != CACHE_SHA
            or meta.get("array_sha256") != CACHE_SHA
            or meta.get("checkpoint_sha256") != STAGE143_SHA
            or meta.get("split") != "validation"):
        raise ValueError("Unexpected Stage143 validation cache")
    old = np.load(args.stage143_cache).astype(np.float64)
    compact_sha = sha(args.compact)
    payload = torch.load(args.compact, map_location="cpu", weights_only=True)
    config = json.loads((ROOT / "configs/stage159_compact_complement_rdrop.json").read_text())
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_rdrop"
            or payload.get("config") != config
            or payload.get("seed") != 17
            or payload.get("train_tokens") != 2400 * 32 * 256):
        raise ValueError("Unexpected Stage159 pilot checkpoint ancestry")
    device, _ = setup("cuda", "fp32", 4)
    model, module_sha = make_model(payload["implementation"], config, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    tokens, byte_count = load_data()["validation"]
    parts = []
    with torch.inference_mode():
        for x, y in windows(tokens, batch_size=32):
            x, y = x.to(device), y.to(device)
            logp = model.predict_log_probs(x).float()
            target = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            parts.append(target[y != -100].double().cpu().numpy())
    new = np.concatenate(parts)
    if (len(old) != 376599 or len(new) != len(old) or byte_count != 1148007
            or not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError("Incomplete validation target stream")
    scale = math.log(2) * byte_count
    old_bpb = -float(old.sum()) / scale
    if abs(old_bpb - STAGE143_BPB) > 2e-5:
        raise ValueError("Stage143 cache BPB changed")
    rows = []
    for weight in WEIGHTS:
        mixed = old if weight == 0 else np.logaddexp(
            old + math.log1p(-weight), new + math.log(weight))
        rows.append(dict(weight=weight, bpb=-float(mixed.sum()) / scale,
                         nll_nats=-float(mixed.sum())))
    best_nonzero = min(rows[1:], key=lambda row: row["bpb"])
    result = dict(
        protocol=PROTOCOL, purpose="target_probability_validation_diagnostic_only",
        split="validation", precision="fp32", targets=len(new),
        utf8_bytes=byte_count, stage143_checkpoint_sha256=STAGE143_SHA,
        stage143_cache_sha256=CACHE_SHA,
        compact_checkpoint_sha256=compact_sha,
        compact_module_sha256=module_sha, source_sha256=sha(Path(__file__)),
        compact_individual_bpb=-float(new.sum()) / scale,
        fixed_mixture_rows=rows, best_nonzero=best_nonzero,
        gain_vs_stage143_bpb=old_bpb - best_nonzero["bpb"],
        full_training_gate_passed=(old_bpb - best_nonzero["bpb"] >= .005),
        no_test_scoring=True,
        warning="Target-only diagnostic; no deployable combination or resource claim",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
