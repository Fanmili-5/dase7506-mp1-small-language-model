"""Reproduce Stage143 validation without opening the test text.

The official ``evaluate.score`` function is used unchanged. This wrapper only
selects the validation bytes explicitly because ``common.load_data`` eagerly
opens all three splits even when the caller requests validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from common import PROTOCOL, make_model, setup, sha  # noqa: E402
from evaluate import score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    if args.threads < 1:
        parser.error("--threads must be positive")

    manifest = json.loads((CODE / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Protocol mismatch")
    tokenizer_path = CODE / "data/tokenizer.json"
    validation_path = CODE / "data/wikitext_validation.txt"
    for name, path in (("tokenizer.json", tokenizer_path),
                       ("wikitext_validation.txt", validation_path)):
        if sha(path) != manifest["sha256"][name]:
            raise ValueError(f"Changed supplied file: {name}")
    raw_validation = validation_path.read_bytes()
    tokens = torch.tensor(
        Tokenizer.from_file(str(tokenizer_path)).encode(
            raw_validation.decode("utf-8")
        ).ids,
        dtype=torch.long,
    )

    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Checkpoint protocol mismatch")
    model, implementation_sha = make_model(
        checkpoint["implementation"], checkpoint["config"], device
    )
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        result = score(model, tokens, len(raw_validation), device, precision)
    losses = result.pop("window_nll_nats")
    if len(losses) != math.ceil((len(tokens) - 1) / 256):
        raise ValueError("Incomplete validation-window coverage")
    if result["targets"] != 376_599 or result["utf8_bytes"] != 1_148_007:
        raise ValueError("Unexpected validation coverage")

    result.update(
        status="stage143_validation_only_reboot_reproduction",
        protocol=PROTOCOL,
        split="validation",
        device="cpu",
        precision=precision,
        threads=args.threads,
        windows=len(losses),
        checkpoint_sha256=sha(args.checkpoint),
        implementation_sha256=implementation_sha,
        evaluator_sha256=sha(CODE / "evaluate.py"),
        tokenizer_sha256=sha(tokenizer_path),
        validation_sha256=sha(validation_path),
        test_text_opened_by_this_script=False,
        method_frozen=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
