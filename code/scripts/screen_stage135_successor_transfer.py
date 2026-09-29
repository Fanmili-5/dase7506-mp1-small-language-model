"""Score a no-training successor-value swap under the fixed MP1 evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save

STAGE115_SHA = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"
IMPLEMENTATION = "student_stage135_successor_gate"
SOURCE_FILES = (
    "student_stage135_successor_gate.py", "student_stage115_order5_gate.py",
    "student_stage105_gated_singlepass.py", "student_stage104_gated_fast.py",
    "student_stage103_gated.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "common.py", "evaluate.py", "data/manifest.json", "data/tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if sha(args.checkpoint) != STAGE115_SHA:
        raise ValueError("Stage115 checkpoint hash mismatch")
    device, _ = setup("cpu", "fp32", 4)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if parent["protocol"] != PROTOCOL or parent["implementation"] != "student_stage115_order5_gate":
        raise ValueError("Unexpected Stage115 payload")
    model, implementation_sha = make_model(IMPLEMENTATION, parent["config"], device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    ids, _ = next(windows(load_data()["validation"][0], 1))
    changed = ids.clone()
    changed[:, 17:] = (changed[:, 17:] + 1) % 2048
    with torch.inference_mode():
        full = model.predict_log_probs(ids)
        changed_prefix = model.predict_log_probs(changed)[:, :17]
        short = model.predict_log_probs(ids[:, :17])
    max_norm_error = float(full.logsumexp(-1).abs().max())
    max_future_error = float((full[:, :17] - changed_prefix).abs().max())
    max_short_error = float((full[:, :17] - short).abs().max())
    if (not torch.isfinite(full).all() or max_norm_error > 1e-5
            or max_future_error > 1e-5 or max_short_error > 1e-5):
        raise ValueError("Successor model causal/normalization smoke failed")
    args.run_dir.mkdir(parents=True)
    output_checkpoint = args.run_dir / "stage135-successor-transfer.pt"
    exported = dict(parent)
    exported["implementation"] = IMPLEMENTATION
    exported["ancestry"] = dict(source_checkpoint_sha256=STAGE115_SHA,
                                 inference_value_alignment_only=True,
                                 no_validation_gradient_updates=True,
                                 no_test_scoring=True)
    atomic_torch_save(exported, output_checkpoint)
    output = args.run_dir / "validation-gpu-fp32.json"
    subprocess.run([sys.executable, "evaluate.py", "--checkpoint",
                    str(output_checkpoint.resolve()), "--device", "cuda",
                    "--precision", "fp32", "--threads", "4", "--split",
                    "validation", "--output", str(output.resolve())],
                   cwd=ROOT, check=True)
    measured = json.loads(output.read_text(encoding="utf-8"))
    if measured["targets"] != 376599 or measured["utf8_bytes"] != 1148007:
        raise ValueError("Validation coverage mismatch")
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    result = dict(protocol=PROTOCOL, stage115_sha256=STAGE115_SHA,
                  exported_checkpoint_sha256=sha(output_checkpoint),
                  implementation_sha256=implementation_sha,
                  script_sha256=sha(Path(__file__)), source_hashes=source_hashes,
                  max_log_normalization_error=max_norm_error,
                  max_future_prefix_log_error=max_future_error,
                  max_short_prefix_log_error=max_short_error,
                  validation_gpu_fp32_bpb=measured["bpb"],
                  validation_targets=measured["targets"],
                  passes_quality_gate=measured["bpb"] < 1.4,
                  cpu_resource_not_yet_qualified=True, no_test_scoring=True)
    atomic_json_dump(result, args.run_dir / "result.json")
    print(json.dumps(result | {"source_hashes": "recorded_in_result_json"}, indent=2))


if __name__ == "__main__":
    main()
