# Installation and evaluation

Commands run from this `code/` directory. Use Python 3.12 and an x86-64 Windows or Linux CPU for the submitted OpenVINO predictor. The measurements are recorded in [RESULTS.md](../RESULTS.md); timing is machine-dependent. The submitted runtime has not been validated on Apple Silicon.

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

Extract the matching checkpoint ZIP at the repository root. It supplies `checkpoints/stage143-openvino-order6.pt`, `inference_assets/stage143-stage92-features.onnx` and their exact inference source files. The supplied data are included in the repository. After installing dependencies and obtaining the bundle, scoring works offline.

## Verify and score

```bash
python scripts/verify_fixed_files.py
python scripts/verify_submission.py
python -m unittest discover -s tests -v
python evaluate.py --checkpoint checkpoints/stage143-openvino-order6.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-validation.json
python evaluate.py --checkpoint checkpoints/stage143-openvino-order6.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

Expected validation BPB: 1.399686162042141. Recorded test BPB: 1.415657616535609. Floating-point differences may depend on runtime and CPU. Do not use test to select or tune a model.

| Split | Scored targets | Raw UTF-8 bytes |
| --- | ---: | ---: |
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

The evaluator scores independent 256-token causal windows, including the final short window. BPB is total negative log-base-2 probability divided by all raw UTF-8 bytes in the split. It is not token perplexity.

## Resource check

The retained timing-control checkpoint has the supplied baseline architecture. Its longer training duration distinguishes it from the initial 1,200-update accuracy baseline but does not change its inference graph.

```bash
python scripts/benchmark_cpu.py --baseline benchmark_controls/baseline-stage3-long-s17.pt --candidate checkpoints/stage143-openvino-order6.pt --threads 4 --repeats 3 --output resource-reproduction.json
```

This repeats scoring of the already-frozen model in fresh processes. Limits are 5× baseline median scoring time, 4 GiB peak whole-process RSS and 64 MiB uncompressed inference assets. `verify_submission.py` checks the complete 55,810,412-byte inference file set, including graph and source.

## Train

See [TRAINING.md](../TRAINING.md) for the final recipe and controlled comparisons. For the original course baseline:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --steps 1200 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output runs/baseline/validation.json
```

Run directories must be new. Fresh training/export stays separate from the frozen release.

## Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8. The tokenizer is fitted only to training text; hashes are in `data/manifest.json`. These notices do not assign a new license to the classroom code.
