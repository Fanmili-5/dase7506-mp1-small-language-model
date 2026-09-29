"""Fresh-process CPU FP32 resource probe that opens validation text only.

This is an ancillary post-reboot check. The unchanged course score function
and whole-process peak-memory function are reused; the provided evaluator and
its eager all-split loader remain unmodified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from common import PROTOCOL, make_model, setup, sha  # noqa: E402
from evaluate import score  # noqa: E402
from peak_memory import peak_process_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((CODE / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Protocol mismatch")
    tokenizer_path = CODE / "data/tokenizer.json"
    validation_path = CODE / "data/wikitext_validation.txt"
    for path in (tokenizer_path, validation_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Changed supplied file: {path.name}")
    raw = validation_path.read_bytes()
    tokens = torch.tensor(
        Tokenizer.from_file(str(tokenizer_path)).encode(raw.decode("utf-8")).ids,
        dtype=torch.long,
    )
    device, precision = setup("cpu", "fp32", 4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Checkpoint protocol mismatch")
    model, implementation_sha = make_model(
        checkpoint["implementation"], checkpoint["config"], device
    )
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        result = score(model, tokens, len(raw), device, precision)
    window_nll = result.pop("window_nll_nats")
    if len(window_nll) != 1472 or result["targets"] != 376_599:
        raise ValueError("Incomplete validation coverage")
    if result["utf8_bytes"] != 1_148_007:
        raise ValueError("Unexpected validation byte count")
    result.update(
        protocol=PROTOCOL,
        split="validation",
        precision=precision,
        threads=4,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha(args.checkpoint),
        checkpoint_bytes=args.checkpoint.stat().st_size,
        implementation_sha256=implementation_sha,
        evaluator_sha256=sha(CODE / "evaluate.py"),
        tokenizer_sha256=sha(tokenizer_path),
        validation_sha256=sha(validation_path),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        platform=platform.platform(),
        torch_version=str(torch.__version__),
        test_text_opened_by_this_script=False,
        **peak_process_memory(),
    )
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
