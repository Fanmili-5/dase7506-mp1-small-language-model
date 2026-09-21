"""Audit the environment without training or evaluating test predictions.

The fixed load_data helper verifies and tokenizes all supplied splits, including
test. This script only reports train/validation sizes and never scores test.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_data, make_model, sha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/student_control.json")
    parser.add_argument("--implementation", default="student")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, implementation_sha = make_model(args.implementation, config, torch.device("cpu"))
    data = load_data()
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "config": str(args.config),
        "config_sha256": sha(args.config),
        "implementation_sha256": implementation_sha,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fp32_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "train_tokens_in_dataset": int(data["train"][0].numel()),
        "validation_tokens_in_dataset": int(data["validation"][0].numel()),
        "test_text_not_evaluated": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
