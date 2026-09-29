"""Hindsight-only window routing upper bound for frozen Stage143/155."""
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
from train_experiment import atomic_json_dump

STAGE143_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
STAGE143_CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
STAGE155_SHA = "c2fe32ba15b0f13b11b43247f058971dac5717ac37fd2acda1bbaa173277bce1"
STAGE143_BPB = 1.399686162042141
STAGE155_BPB = 1.409877270416759
HALF_MIXTURE_BPB = 1.3761826642
FRACTIONS = (0.10, 0.20, 0.30, 0.35, 0.50, 1.0)


def oracle_table(old: np.ndarray, new: np.ndarray, byte_count: int) -> dict:
    """Return an unattainable hindsight upper bound at fixed window budgets."""
    if old.shape != new.shape or len(old) != 376599:
        raise ValueError("Incomplete paired target streams")
    mixture = np.logaddexp(old, new) - math.log(2)
    n_windows = math.ceil(len(old) / 256)
    improvements = np.asarray([
        float((mixture[start:start + 256] - old[start:start + 256]).sum())
        for start in range(0, len(old), 256)], dtype=np.float64)
    if len(improvements) != n_windows:
        raise ValueError("Independent window mapping mismatch")
    # Log-probability improvement is old NLL minus mixed NLL. A hindsight
    # oracle uses only positive cells and never spends budget on a bad window.
    positive = np.sort(improvements[improvements > 0])[::-1]
    scale = math.log(2) * byte_count
    original_bpb = -float(old.sum()) / scale
    table = {}
    for fraction in FRACTIONS:
        allowance = math.floor(fraction * n_windows)
        selected = min(allowance, len(positive))
        gain = float(positive[:selected].sum()) / scale
        table[f"{fraction:.2f}"] = dict(
            max_fraction=fraction, max_windows=allowance,
            oracle_selected_windows=selected,
            oracle_bpb=original_bpb - gain,
            oracle_gain_vs_stage143_bpb=gain)
    return dict(total_independent_windows=n_windows,
                windows_where_mixture_helps=int(len(positive)),
                window_gain_nats_min=float(improvements.min()),
                window_gain_nats_median=float(np.median(improvements)),
                window_gain_nats_max=float(improvements.max()),
                oracle_budget_table=table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage143-cache", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage162 diagnostic directory")
    meta = json.loads(args.stage143_cache.with_suffix(".json").read_text())
    if (sha(args.stage143_cache) != STAGE143_CACHE_SHA
            or meta.get("array_sha256") != STAGE143_CACHE_SHA
            or meta.get("checkpoint_sha256") != STAGE143_SHA
            or meta.get("split") != "validation"
            or sha(args.stage155) != STAGE155_SHA):
        raise ValueError("Unexpected frozen validation diagnostic input")
    old = np.load(args.stage143_cache).astype(np.float64)
    payload = torch.load(args.stage155, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_rdrop"
            or payload.get("seed") != 17
            or payload.get("train_tokens") != 7200 * 32 * 256):
        raise ValueError("Unexpected Stage155 training ancestry")
    device, _ = setup("cuda", "fp32", 4)
    model, module_sha = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    validation, byte_count = load_data()["validation"]
    chunks = []
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            logp = model.predict_log_probs(x.to(device))
            target = logp.gather(-1, y.clamp_min(0).to(device)[..., None]).squeeze(-1)
            chunks.append(target[y.to(device) != -100].double().cpu().numpy())
    new = np.concatenate(chunks)
    if (old.shape != (376599,) or new.shape != old.shape
            or byte_count != 1148007
            or not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError("Incomplete or non-finite validation target streams")
    scale = math.log(2) * byte_count
    old_bpb = -float(old.sum()) / scale
    new_bpb = -float(new.sum()) / scale
    half_bpb = -float((np.logaddexp(old, new) - math.log(2)).sum()) / scale
    if (abs(old_bpb - STAGE143_BPB) > 2e-5
            or abs(new_bpb - STAGE155_BPB) > 2e-5
            or abs(half_bpb - HALF_MIXTURE_BPB) > 2e-5):
        raise ValueError("Frozen complete-validation score mismatch")
    summary = oracle_table(old, new, byte_count)
    args.run_dir.mkdir(parents=True)
    array_path = args.run_dir / "stage155-target-logp.npy"
    np.save(array_path, new)
    result = dict(
        protocol=PROTOCOL, purpose="hindsight_only_oracle_not_deployable",
        split="validation", precision="fp32", device=str(device),
        targets=len(new), utf8_bytes=byte_count,
        stage143_checkpoint_sha256=STAGE143_SHA,
        stage143_cached_array_sha256=STAGE143_CACHE_SHA,
        stage155_checkpoint_sha256=STAGE155_SHA,
        stage155_implementation_sha256=module_sha,
        stage155_target_array_sha256=sha(array_path),
        source_sha256=sha(Path(__file__)),
        stage143_bpb=old_bpb, stage155_bpb=new_bpb,
        unconditional_half_mixture_bpb=half_bpb,
        **summary,
        primary_oracle_fraction=.30,
        primary_oracle_required_bpb=1.33,
        primary_oracle_gate_passed=(summary["oracle_budget_table"]["0.30"]["oracle_bpb"] <= 1.33),
        warning="Validation-label hindsight selection is impossible at inference; first-token-only whole-window routing would be causal but weaker; assets and CPU unqualified",
        no_test_scoring=True)
    atomic_json_dump(result, args.run_dir / "oracle.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
