"""Remove all Stage26 training-only heads and export the original inference graph."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from student_multi_token import inference_config, inference_state
from train_experiment import atomic_torch_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_multi_token":
        raise ValueError("Expected a Stage26 multi-token checkpoint")
    source, _ = make_model("student_multi_token", payload["config"], torch.device("cpu"))
    source.load_state_dict(payload["model"])
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_structured", config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(payload["model"]), strict=True)
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    removed = sorted(set(payload["model"]) - set(deployed.state_dict()))
    exported = dict(
        payload, implementation="student_structured", config=config, model=deployed.state_dict(),
        training_deep_supervision=dict(
            source_checkpoint_sha256=sha(args.checkpoint),
            layers=payload["config"]["deep_supervision_layers"],
            weight=payload["config"]["deep_supervision_weight"],
            same_unique_primary_targets=True,
        ),
        training_multi_token=dict(
            source_implementation_sha256=sha(ROOT / "student_multi_token.py"),
            offsets=payload["config"]["future_prediction_offsets"],
            weight=payload["config"]["future_prediction_weight"],
            removed_state_keys=removed, export_script_sha256=sha(Path(__file__)),
            note="All auxiliary norms/projections removed; deployed tensors and outputs use the original prefix-copy graph.",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
