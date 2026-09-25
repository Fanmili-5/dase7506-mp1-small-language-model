"""Compare fixed Stage143/155 validation errors and one 50:50 mixture."""
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
from tokenizers import Tokenizer

from common import PROTOCOL, load_data, make_model, setup, sha, windows

STAGE143_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
STAGE143_CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
STAGE155_SHA = "c2fe32ba15b0f13b11b43247f058971dac5717ac37fd2acda1bbaa173277bce1"
STAGE143_BPB = 1.399686162042141
STAGE155_BPB = 1.409877270416759


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage143-cache", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic output")
    cache_meta = json.loads(args.stage143_cache.with_suffix(".json").read_text())
    if (sha(args.stage143_cache) != STAGE143_CACHE_SHA
            or cache_meta.get("array_sha256") != STAGE143_CACHE_SHA
            or cache_meta.get("checkpoint_sha256") != STAGE143_SHA
            or cache_meta.get("split") != "validation"
            or sha(args.stage155) != STAGE155_SHA):
        raise ValueError("Unexpected frozen diagnostic input")
    old = np.load(args.stage143_cache).astype(np.float64)
    payload = torch.load(args.stage155, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_rdrop"
            or payload.get("seed") != 17
            or payload.get("train_tokens") != 7200 * 32 * 256):
        raise ValueError("Unexpected Stage155 training ancestry")
    device, _ = setup("cuda", "fp32", 4)
    model, implementation_sha = make_model(payload["implementation"],
                                            payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    validation, byte_count = load_data()["validation"]
    chunks = []
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            logp = model.predict_log_probs(x)
            target = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            chunks.append(target[y != -100].double().cpu().numpy())
    new = np.concatenate(chunks)
    if (len(old) != 376599 or len(new) != len(old) or byte_count != 1148007
            or not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError("Incomplete validation target stream")
    scale = math.log(2) * byte_count
    old_bpb = -float(old.sum()) / scale
    new_bpb = -float(new.sum()) / scale
    if abs(old_bpb - STAGE143_BPB) > 2e-5 or abs(new_bpb - STAGE155_BPB) > 2e-5:
        raise ValueError(f"Frozen BPB mismatch: {old_bpb}, {new_bpb}")
    mixture = np.logaddexp(old, new) - math.log(2)
    mixture_bpb = -float(mixture.sum()) / scale

    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_train.txt", "wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError(f"Fixed data changed: {name}")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_ids = np.asarray(tokenizer.encode(
        (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")).ids)
    validation_ids = np.asarray(tokenizer.encode(
        (ROOT / "data/wikitext_validation.txt").read_text(encoding="utf-8")).ids)
    if len(validation_ids) - 1 != len(new):
        raise ValueError("Validation token alignment changed")
    frequency = np.bincount(train_ids, minlength=2048)
    mid = (frequency[validation_ids[1:]] >= 100) & (frequency[validation_ids[1:]] < 1000)
    seen = np.zeros(len(new), dtype=np.bool_)
    prefix: set[int] = set()
    for index, target in enumerate(validation_ids[1:]):
        if index % 256 == 0:
            prefix.clear()
        prefix.add(int(validation_ids[index]))
        seen[index] = int(target) in prefix
    groups = {}
    for freq_name, freq_mask in (("mid_frequency_100_999", mid),
                                 ("other_frequency", ~mid)):
        for seen_name, seen_mask in (("seen_prefix", seen),
                                     ("unseen_prefix", ~seen)):
            mask = freq_mask & seen_mask
            groups[f"{freq_name}/{seen_name}"] = {
                "targets": int(mask.sum()),
                "stage143_nll_nats": -float(old[mask].sum()),
                "stage155_nll_nats": -float(new[mask].sum()),
                "mixture_nll_nats": -float(mixture[mask].sum()),
                "stage155_minus_stage143_nll_nats": float((old[mask] - new[mask]).sum()),
            }
    result = {
        "protocol": PROTOCOL,
        "purpose": "diagnostic_only_not_deployable_oracle_or_inference_gate",
        "split": "validation", "precision": "fp32", "device": str(device),
        "stage143_checkpoint_sha256": STAGE143_SHA,
        "stage143_cache_sha256": STAGE143_CACHE_SHA,
        "stage155_checkpoint_sha256": STAGE155_SHA,
        "stage155_implementation_sha256": implementation_sha,
        "source_sha256": sha(Path(__file__)),
        "targets": len(new), "utf8_bytes": byte_count,
        "stage143_bpb": old_bpb, "stage155_bpb": new_bpb,
        "fixed_half_mixture_bpb": mixture_bpb,
        "mixture_gain_vs_stage143_bpb": old_bpb - mixture_bpb,
        "distillation_pilot_gate": mixture_bpb <= 1.385 and old_bpb - mixture_bpb >= .014,
        "groups": groups, "no_test_scoring": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
