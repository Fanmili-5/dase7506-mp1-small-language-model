"""Remove training-only auxiliary norms and export the original inference graph."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import PROTOCOL, make_model, sha
from student_deep_supervision import inference_config, inference_state
from train_experiment import atomic_torch_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_deep_supervision":
        raise ValueError("Expected a deep-supervision checkpoint")
    source, _ = make_model(payload["implementation"], payload["config"], torch.device("cpu"))
    source.load_state_dict(payload["model"])
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_structured", config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(payload["model"]), strict=True)
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2,256) * 29 + 7) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(payload, implementation="student_structured", config=config,
        model=deployed.state_dict(), training_deep_supervision=dict(
            source_checkpoint_sha256=sha(args.checkpoint),
            source_implementation_sha256=sha(ROOT / "student_deep_supervision.py"),
            layers=payload["config"]["deep_supervision_layers"],
            weight=payload["config"]["deep_supervision_weight"],
            same_unique_next_token_targets=True,
            removed_state_keys=sorted(set(payload["model"])-set(deployed.state_dict())),
            export_script_sha256=sha(Path(__file__)),
            note="Auxiliary norms removed; deployed tensors/output match the original prefix-copy graph."))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
