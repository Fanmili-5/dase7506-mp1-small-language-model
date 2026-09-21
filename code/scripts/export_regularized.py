"""Export identical eval weights into the original StructuredLM implementation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import PROTOCOL, sha, make_model
from student_regularized import inference_config, REGULARIZATION_KEYS
from train_experiment import atomic_torch_save


def export_payload(payload, source_sha):
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_regularized":
        raise ValueError("Expected a regularized MP1 checkpoint")
    source, _ = make_model("student_regularized", payload["config"], torch.device("cpu"))
    config = inference_config(payload["config"])
    deployed, _ = make_model("student_structured", config, torch.device("cpu"))
    source.load_state_dict(payload["model"])
    deployed.load_state_dict(payload["model"], strict=True)
    source.eval()
    deployed.eval()
    # Synthetic IDs only for output-equivalence QA, no training or score fitting.
    ids = (torch.arange(512).reshape(2, 256) * 17) % 2048
    with torch.no_grad():
        torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)
    result = dict(payload)
    result.update(implementation="student_structured", config=config,
        training_regularization=dict(source_checkpoint_sha256=source_sha,
            source_implementation_sha256=sha(ROOT / "student_regularized.py"),
            probabilities={k: payload["config"].get(k, 0) for k in REGULARIZATION_KEYS},
            export_script_sha256=sha(Path(__file__)),
            note="No weight changes. Training-only masks removed. Full-context FP32 export parity checked."))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    result = export_payload(torch.load(args.checkpoint, map_location="cpu", weights_only=True), sha(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(result, args.output)
    print(f"Exported identical predictor: {args.output}; sha256={sha(args.output)}")


if __name__ == "__main__":
    main()
