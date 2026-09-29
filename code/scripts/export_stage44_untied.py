"""Remove Stage44 training-only heads while preserving untied embeddings."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from student_multi_token import inference_config, inference_state
from train_experiment import atomic_torch_save


def materialize_model(config, state):
    source, _ = make_model("student_untied_multi_token", config, torch.device("cpu"))
    source.load_state_dict(state)
    deployed_config = inference_config(config)
    deployed, _ = make_model("student_untied_structured", deployed_config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(state), strict=True)
    return source, deployed, deployed_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_untied_multi_token":
        raise ValueError("Expected a Stage44 untied multi-token checkpoint")
    source, deployed, config = materialize_model(payload["config"], payload["model"])
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(
        payload, implementation="student_untied_structured", config=config,
        model=deployed.state_dict(),
        training_untied_output=dict(
            source_checkpoint_sha256=sha(args.checkpoint),
            output_weight_is_independent=True,
            export_script_sha256=sha(Path(__file__))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
