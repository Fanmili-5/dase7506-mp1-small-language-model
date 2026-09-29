"""Select the prespecified Stage197 endpoint or last-five average on validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup, sha
from evaluate import score
from scripts.average_checkpoints import average_checkpoints
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import atomic_json_dump, atomic_torch_save

STEPS = (6000, 6300, 6600, 6900, 7200)
TARGETS_PER_STEP = 32 * 256


def score_checkpoint(payload: dict, data: dict, device: torch.device) -> dict:
    model, implementation_sha = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with torch.no_grad():
        result = score(model, *data["validation"], device, "fp32", 32)
    result.pop("window_nll_nats")
    if result["targets"] != 376_599 or result["utf8_bytes"] != 1_148_007:
        raise ValueError("Incomplete or changed validation coverage")
    result["implementation_sha256"] = implementation_sha
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8-sig"))
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    if (run.get("protocol") != PROTOCOL or run.get("seed") != 17
            or run.get("steps") != 7200
            or run.get("primary_targets") != 58_982_400
            or metrics.get("status") != "completed_training_validation_only"):
        raise ValueError("Stage197 full run is not complete and matched")
    for relative, expected in run["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Training source changed: {relative}")
    paths = [args.run_dir / "checkpoints" / f"step-{step:06d}.pt" for step in STEPS]
    for step, path in zip(STEPS, paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("protocol") != PROTOCOL
                or payload.get("implementation") != "student_stage153_shared_branch"
                or payload.get("seed") != 17
                or payload.get("train_tokens") != step * TARGETS_PER_STEP):
            raise ValueError(f"Unexpected trajectory checkpoint: {path}")
    averaged = average_checkpoints(paths)
    final = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (final.get("implementation") != "student_stage153_shared_branch"
            or final.get("seed") != 17
            or final.get("train_tokens") != 58_982_400):
        raise ValueError("Unexpected Stage197 endpoint checkpoint")
    args.output_dir.mkdir(parents=True)
    average_path = args.output_dir / "last-five-average.pt"
    atomic_torch_save(averaged, average_path)
    data = load_train_validation()
    device, precision = setup("cuda", "fp32", 4)
    endpoint_score = score_checkpoint(final, data, device)
    average_score = score_checkpoint(averaged, data, device)
    selected = "endpoint" if endpoint_score["bpb"] <= average_score["bpb"] else "last_five_average"
    result = {
        "status": "complete_validation_selection_only", "protocol": PROTOCOL,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "precision": precision,
        "selected": selected,
        "selected_bpb": min(endpoint_score["bpb"], average_score["bpb"]),
        "endpoint_validation": endpoint_score,
        "average_validation": average_score,
        "endpoint_checkpoint_sha256": sha(args.run_dir / "checkpoint.pt"),
        "average_checkpoint_sha256": sha(average_path),
        "source_checkpoint_sha256": [sha(path) for path in paths],
        "average_steps": list(STEPS),
        "stage143_validation_bpb": 1.399686162042141,
        "advancement_gate_passed": min(endpoint_score["bpb"], average_score["bpb"]) <= 1.384686162042141,
        "validation_goal_met": min(endpoint_score["bpb"], average_score["bpb"]) < 1.35,
        "no_test_file_opened": True,
    }
    atomic_json_dump(result, args.output_dir / "selection.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
