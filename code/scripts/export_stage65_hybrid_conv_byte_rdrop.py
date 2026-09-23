"""Fold Stage65 byte composition into the standard hybrid-conv inference model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from student_hybrid_conv_byte_rdrop import inference_config, inference_state
from train_experiment import atomic_torch_save


def materialize_model(config, source_state):
    source, _ = make_model(
        "student_hybrid_conv_byte_rdrop", config, torch.device("cpu")
    )
    source.load_state_dict(source_state, strict=True)
    deployed_config = inference_config(config)
    state = inference_state(source_state)
    with torch.no_grad():
        materialized = source.composed_weight().detach().clone()
    state["token.weight"] = materialized
    state["head.weight"] = materialized
    deployed, _ = make_model(
        "student_hybrid_conv_structured", deployed_config, torch.device("cpu")
    )
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
            or payload["implementation"] != "student_hybrid_conv_byte_rdrop"):
        raise ValueError("Expected a Stage65 hybrid-conv byte/R-Drop checkpoint")
    source, deployed, config = materialize_model(payload["config"], payload["model"])
    source.eval(); deployed.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    exported = dict(
        payload,
        implementation="student_hybrid_conv_structured",
        config=config,
        model=deployed.state_dict(),
        training_byte_rdrop=dict(
            source_checkpoint_sha256=sha(args.checkpoint),
            feature_kind=payload["config"]["byte_features"],
            projection_shape=list(source.byte_projection.shape),
            rdrop_alpha=payload["config"]["rdrop_alpha"],
            export_script_sha256=sha(Path(__file__)),
            note="Training-only byte projection materialized into tied token/output weights.",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(exported, args.output)
    print(f"Exported {args.output}; sha256={sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
