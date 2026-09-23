"""Fold Stage69 output LoRA into one untied inference projection."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from train_experiment import atomic_torch_save


def materialize_model(config, source_state):
    source, _ = make_model("student_hybrid_conv_output_lora", config,
                           torch.device("cpu"))
    source.load_state_dict(source_state, strict=True)
    deployed_config = dict(config)
    deployed_config.pop("output_lora_rank")
    deployed_config.pop("output_lora_alpha")
    deployed_config["untied_output"] = True
    state = dict(source_state)
    state.pop("output_lora_a"); state.pop("output_lora_b")
    with torch.no_grad():
        state["head.weight"] = source.output_weight().detach().clone()
    deployed, _ = make_model("student_hybrid_conv_untied_bias", deployed_config,
                             torch.device("cpu"))
    deployed.load_state_dict(state, strict=True)
    return source, deployed, deployed_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (payload["protocol"] != PROTOCOL
            or payload["implementation"] != "student_hybrid_conv_output_lora"):
        raise ValueError("Expected a Stage69 output-LoRA checkpoint")
    source, deployed, config = materialize_model(payload["config"], payload["model"])
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(
        payload, implementation="student_hybrid_conv_untied_bias",
        config=config, model=deployed.state_dict(),
        training_output_lora=dict(
            source_checkpoint_sha256=sha(args.checkpoint),
            rank=payload["config"]["output_lora_rank"],
            alpha=payload["config"]["output_lora_alpha"],
            export_script_sha256=sha(Path(__file__)),
            note="Training-only low-rank residual folded into one untied output matrix.",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
