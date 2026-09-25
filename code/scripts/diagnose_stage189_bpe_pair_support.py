"""One fixed-alpha, full-validation support diagnostic for frozen Stage143.

The mask is derived exclusively from the supplied tokenizer's direct BPE
merges. This diagnostic does not load test, fit a parameter, or change assets.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from audit_stage189_bpe_merge_pairs import (
    EXPECTED_TOKENIZER_SHA256, direct_merge_pairs,
)


CHECKPOINT_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
GRAPH_SHA256 = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
REFERENCE_BPB = 1.399686162042141
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007
EXPECTED_WINDOWS = 1_472
ALPHA = 0.9


def validation_ids() -> tuple[torch.Tensor, int]:
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unexpected protocol")
    token_path = data / "tokenizer.json"
    val_path = data / "wikitext_validation.txt"
    if (sha(token_path) != EXPECTED_TOKENIZER_SHA256
            or manifest["sha256"]["tokenizer.json"] != EXPECTED_TOKENIZER_SHA256
            or sha(val_path) != manifest["sha256"][val_path.name]):
        raise ValueError("Fixed validation/tokenizer hash mismatch")
    raw = val_path.read_bytes()
    tokenizer = Tokenizer.from_file(str(token_path))
    ids = tokenizer.encode(raw.decode("utf-8")).ids
    if len(raw) != EXPECTED_BYTES or len(ids) - 1 != EXPECTED_TARGETS:
        raise ValueError("Unexpected validation coverage")
    return torch.tensor(ids, dtype=torch.long), len(raw)


def merge_mask() -> torch.Tensor:
    path = ROOT / "data/tokenizer.json"
    pairs = direct_merge_pairs(json.loads(path.read_text(encoding="utf-8")))
    mask = torch.zeros((2048, 2048), dtype=torch.bool)
    for left, right in pairs:
        mask[left, right] = True
    if int(mask.sum()) != 1792:
        raise ValueError("Incorrect direct-merge mask")
    return mask


def adjusted_log_target(logp: torch.Tensor, x: torch.Tensor,
                        y: torch.Tensor, mask: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return baseline/adjusted target logp, removed mass, true-mask flags."""
    if (logp.shape != (*x.shape, 2048) or x.shape != y.shape
            or mask.shape != (2048, 2048)):
        raise ValueError("Unexpected prediction shape")
    forbidden = mask[x]
    p = logp.exp()
    removed = (p * forbidden).sum(-1)
    denominator = 1 - ALPHA * removed
    if (not torch.isfinite(denominator).all() or (denominator <= 0).any()
            or (denominator > 1 + 1e-5).any()):
        raise ValueError("Invalid normalization denominator")
    target_ids = y.clamp_min(0).unsqueeze(-1)
    baseline = logp.gather(-1, target_ids).squeeze(-1)
    target_forbidden = forbidden.gather(-1, target_ids).squeeze(-1)
    adjusted = (baseline.double()
                + torch.log1p(-ALPHA * target_forbidden.double())
                - denominator.double().log())
    if not torch.isfinite(adjusted).all():
        raise ValueError("Invalid adjusted target log probability")
    return baseline.double(), adjusted, removed.double(), target_forbidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Use a new output path and positive threads")
    if sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Stage143 checkpoint changed")
    graph_path = ROOT / "inference_assets/stage143-stage92-features.onnx"
    if sha(graph_path) != GRAPH_SHA256:
        raise ValueError("Stage143 graph changed")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("implementation") != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected Stage143 checkpoint")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    ids, byte_count = validation_ids()
    mask = merge_mask()

    half_nll = [[0.0, 0.0], [0.0, 0.0]]
    half_targets = [0, 0]
    half_masked_true = [0, 0]
    half_removed_mass = [0.0, 0.0]
    maximum_mass = 0.0
    maximum_normalization_error = 0.0
    windows_seen = 0
    batches = 0
    with torch.inference_mode():
        for x, y in windows(ids, batch_size=32):
            logp = model.predict_log_probs(x)
            baseline, adjusted, removed, true_mask = adjusted_log_target(
                logp, x, y, mask)
            valid = y != -100
            full_adjusted = (logp.exp() * (1 - ALPHA * mask[x])
                             / (1 - ALPHA * removed.float()).unsqueeze(-1))
            normal_error = (full_adjusted.sum(-1) - 1).abs().max().item()
            maximum_normalization_error = max(maximum_normalization_error, normal_error)
            if normal_error > 3e-5:
                raise ValueError("Adjusted distribution is not normalized")
            if windows_seen < EXPECTED_WINDOWS // 2 < windows_seen + x.shape[0]:
                raise ValueError("Batch crosses fixed half-window boundary")
            half = 0 if windows_seen < EXPECTED_WINDOWS // 2 else 1
            half_nll[half][0] -= baseline[valid].sum().item()
            half_nll[half][1] -= adjusted[valid].sum().item()
            half_targets[half] += int(valid.sum())
            half_masked_true[half] += int(true_mask[valid].sum())
            half_removed_mass[half] += removed[valid].sum().item()
            maximum_mass = max(maximum_mass, removed[valid].max().item())
            windows_seen += x.shape[0]
            batches += 1
            if batches % 8 == 0:
                print(f"validated {windows_seen}/{EXPECTED_WINDOWS} windows", flush=True)
    if windows_seen != EXPECTED_WINDOWS or sum(half_targets) != EXPECTED_TARGETS:
        raise ValueError("Incomplete validation")
    scale = math.log(2) * byte_count
    reference = sum(row[0] for row in half_nll) / scale
    adjusted_bpb = sum(row[1] for row in half_nll) / scale
    if abs(reference - REFERENCE_BPB) > 2e-5:
        raise ValueError("Unchanged Stage143 score not reproduced")
    half_gain_nats = [row[0] - row[1] for row in half_nll]
    gain = reference - adjusted_bpb
    advance = (gain >= 0.015 and all(value > 0 for value in half_gain_nats)
               and sum(half_masked_true) == 0)
    result = {
        "protocol": PROTOCOL,
        "purpose": "stage189_fixed_direct_merge_support_diagnostic_only",
        "split": "validation", "device": str(device), "precision": precision,
        "threads": args.threads, "checkpoint_sha256": CHECKPOINT_SHA256,
        "graph_sha256": GRAPH_SHA256, "implementation_sha256": implementation_sha,
        "source_sha256": sha(Path(__file__)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "alpha": ALPHA, "mask_pairs": int(mask.sum()),
        "windows": windows_seen, "batches": batches,
        "targets": sum(half_targets), "utf8_bytes": byte_count,
        "half_targets": half_targets,
        "half_masked_true_occurrences": half_masked_true,
        "half_mean_removed_mass": [half_removed_mass[i] / half_targets[i]
                                   for i in range(2)],
        "maximum_removed_mass": maximum_mass,
        "maximum_normalization_error": maximum_normalization_error,
        "reference_bpb": reference, "adjusted_bpb": adjusted_bpb,
        "gain_bpb": gain,
        "half_gain_nats": half_gain_nats,
        "advance_to_inference_qualification": advance,
        "no_test_access_or_scoring": True,
        "warning": "Validation diagnostic only; not a resource-qualified predictor.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
