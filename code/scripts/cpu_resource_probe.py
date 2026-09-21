"""Score one checkpoint on validation in a fresh CPU process with peak RAM."""
import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score
from peak_memory import peak_process_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Unexpected checkpoint protocol")
    model, implementation_sha = make_model(checkpoint["implementation"], checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    data = load_data()
    result = score(model, *data["validation"], device, precision)
    result.pop("window_nll_nats")
    result.update(
        protocol=PROTOCOL, split="validation", precision=precision, threads=args.threads,
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        checkpoint_bytes=args.checkpoint.stat().st_size,
        implementation_sha256=implementation_sha, evaluator_sha256=sha(ROOT / "evaluate.py"),
        tokenizer_sha256=sha(ROOT / "data/tokenizer.json"),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        platform=platform.platform(), torch_version=str(torch.__version__),
        **peak_process_memory(),
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
