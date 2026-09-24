"""Profile Stage105 on one fixed validation input batch without reading targets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import load_data, make_model, setup, windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x, _ = next(windows(load_data()["validation"][0], 32))
    with torch.inference_mode():
        for _ in range(2):
            model.predict_log_probs(x)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True) as profile:
            model.predict_log_probs(x)
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    print(profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=35))


if __name__ == "__main__":
    main()
