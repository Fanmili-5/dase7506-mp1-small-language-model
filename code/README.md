# Installation and evaluation

Run these commands from `code/`. I evaluated the OpenVINO model with Python 3.12 on x86-64 Windows and Linux CPUs; I have not validated it on Apple Silicon. Measurements are in [RESULTS.md](../RESULTS.md).

## Install

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements_stage143.txt
```

For NVIDIA training, replace the CPU PyTorch command with:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Extract the matching checkpoint ZIP at the repository root. It supplies `code/checkpoints/final-model.pt`, `code/inference_assets/stage143-stage92-features.onnx` and their exact inference source files. The repository includes the supplied data. After installation and download, scoring works offline.

## Verify and score

```bash
python scripts/verify_fixed_files.py
python scripts/verify_submission.py
python -m unittest discover -s tests -v
python evaluate.py --checkpoint checkpoints/final-model.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-validation.json
python evaluate.py --checkpoint checkpoints/final-model.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

Expected validation BPB: 1.3996861637466336. Recorded test BPB: 1.4156576308174311 ([test result](results/final-evidence/test.json)). Small floating-point differences can depend on runtime and CPU. The course leaderboard uses full-test FP32 BPB; validation is for model selection.

The runtime checks FP32 execution and caps worker count at the requested number of threads. It may use fewer workers on a limited host. The evaluator defaults to four threads, matching the recorded measurements. Compare both models on the same host with the same thread count and workload.

| Split | Scored targets | Raw UTF-8 bytes |
| --- | ---: | ---: |
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

The evaluator scores independent 256-token causal windows, including the final short window. BPB is total negative log-base-2 probability divided by all raw UTF-8 bytes in the split. It is not token perplexity.

## Resource check

The retained timing-control checkpoint has the supplied baseline architecture. Its longer training duration distinguishes it from the initial 1,200-update accuracy baseline but does not change its inference graph.

```bash
python scripts/benchmark_cpu.py --baseline benchmark_controls/baseline-stage3-long-s17.pt --candidate checkpoints/final-model.pt --threads 4 --repeats 3 --output resource-reproduction.json
```

This repeats scoring in fresh processes and reports the ratio of median times. The course limits are 5× baseline CPU scoring time, 4 GiB peak evaluation RSS and 64 MiB uncompressed inference assets. `verify_submission.py` checks the model files, including the graph and required source, and reports their total byte count.

The recorded Linux rerun gave 4.643160× and 2.14 GiB peak RSS. An earlier run of the same files gave 5.360158×; both are retained in [RESULTS.md](../RESULTS.md). Hardware and runtime scheduling affect relative speed, so these are measurements on specific hosts rather than a universal timing guarantee.

## Train

See [TRAINING.md](../TRAINING.md) for the training procedure and controlled comparisons. For the original course baseline:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --steps 1200 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output runs/baseline/validation.json
```

Use a new run directory for each training run; outputs do not overwrite the downloadable checkpoint.

## Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8. The tokenizer is fitted only to training text; hashes are in `data/manifest.json`. These notices do not assign a new license to the classroom code.

## Directory layout

`checkpoints/` contains the model weights, and `inference_assets/` contains the neural graph. `benchmark_controls/` holds the baseline used for timing. `results/` contains evaluation results and comparison/cost summaries. Training and evaluation utilities are in `scripts/`; correctness tests are in `tests/`.
