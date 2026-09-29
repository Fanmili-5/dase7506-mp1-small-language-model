"""Remove Stage48 training-only heads while preserving top-1 MoE inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from student_moe_multi_token import inference_config, inference_state
from train_experiment import atomic_torch_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_moe_multi_token":
        raise ValueError("Expected a Stage48 MoE checkpoint")
    source, _ = make_model("student_moe_multi_token", payload["config"], torch.device("cpu"))
    source.load_state_dict(payload["model"])
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_moe_structured", config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(payload["model"]), strict=True)
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(payload, implementation="student_moe_structured", config=config,
                    model=deployed.state_dict(), training_moe=dict(
                        source_checkpoint_sha256=sha(args.checkpoint),
                        balance_weight=payload["config"]["moe_balance_weight"],
                        z_weight=payload["config"]["moe_z_weight"],
                        export_script_sha256=sha(Path(__file__))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
