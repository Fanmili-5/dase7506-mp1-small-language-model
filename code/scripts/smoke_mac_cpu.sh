#!/usr/bin/env bash
set -euo pipefail

code_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$code_root"

python_bin="${PYTHON_BIN:-$code_root/.venv/bin/python}"
stamp="$(date +%Y%m%d-%H%M%S)"
run_dir="runs/mac-smoke-$stamp"

"$python_bin" -m unittest discover -s tests -v
"$python_bin" scripts/check_environment.py --config configs/student_modern.json
"$python_bin" train_experiment.py \
  --implementation student \
  --config configs/student_modern.json \
  --device cpu \
  --precision fp32 \
  --steps 2 \
  --micro-batch-size 2 \
  --warmup-steps 1 \
  --log-every 1 \
  --save-every 1 \
  --run-dir "$run_dir"

echo "Mac smoke test complete: $run_dir"
