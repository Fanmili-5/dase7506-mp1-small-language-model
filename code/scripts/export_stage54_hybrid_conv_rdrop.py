"""Remove Stage54 R-Drop/training-only heads into hybrid-conv inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from student_hybrid_conv_rdrop import inference_config, inference_state
from train_experiment import atomic_torch_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_hybrid_conv_rdrop":
        raise ValueError("Expected a Stage54 hybrid-conv R-Drop checkpoint")
    source, _ = make_model(payload["implementation"], payload["config"], torch.device("cpu"))
    source.load_state_dict(payload["model"])
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_hybrid_conv_structured", config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(payload["model"]), strict=True)
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(payload, implementation="student_hybrid_conv_structured", config=config,
                    model=deployed.state_dict(), training_rdrop=dict(
                        source_checkpoint_sha256=sha(args.checkpoint),
                        alpha=payload["config"]["rdrop_alpha"],
                        export_script_sha256=sha(Path(__file__))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
