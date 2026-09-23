"""One final local scalar-calibration expansion after Stage78."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import screen_stage78_stage71_calibration as screen
from common import sha
from train_experiment import atomic_json_dump


# Include both the unchanged Stage71 point and the complete Stage78 winner.
screen.TEMPERATURES = (1., 1.075, 1.10, 1.125, 1.15, 1.175, 1.20)
screen.PRIOR_WEIGHTS = (0., .025, .0375, .05, .0625, .075)
screen.COPY_GATE_SHIFTS = (0., .125, .1875, .25, .3125, .375)
screen.MIXTURE_WEIGHTS = (.05, .0625, .075, .0875, .10, .1125)


if __name__ == "__main__":
    screen.main()
    run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
    output = run_dir / "screen.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    result["purpose"] = "final_bounded_scalar_refinement"
    result["refinement_source_sha256"] = sha(Path(__file__))
    result["scalar_calibration_closed_after_this_scan"] = True
    atomic_json_dump(result, output)
