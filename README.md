# DASE7506 MP1: small language model

This repository contains the course baseline, my experiments, and the model I froze for Project 1. The supplied data, tokenizer and evaluator are unchanged. The model was trained from scratch on the supplied WikiText-2 training split.

## Result

| Model | Full validation BPB | Full test BPB |
| --- | ---: | ---: |
| Course baseline | 2.072081 | 2.102015 |
| Frozen Stage143 model | 1.399686 | 1.415658 |

The test score comes from one CPU FP32 run after the model and inference files were frozen. It is an improvement over the baseline, but it does not reach my hoped-for score below 1.40. The [report](REPORT.pdf) explains the experiments and their limits. The [test result](code/results/stage143-evidence/test-stage143-20260927.json) and [freeze record](code/results/stage143-evidence/freeze-stage143-20260927.json) identify the exact model.

## Model

The neural network has eight causal blocks, alternating global attention with gated causal depthwise convolution. It predicts with a vocabulary head and a copy head restricted to the current window's prefix. A six-gram modified Kneser--Ney model, built only from training text, provides a second distribution. A small gate uses input-derived confidence features to mix them. The evaluator still sees one normalized 2,048-token distribution per position and no state across windows.

Stage143 is the CPU inference packaging of that model, not another training run. Its OpenVINO feature graph stores the backbone once; the checkpoint holds the head, copy mechanism, count tables and gate. The model and training choices are described in the [report](REPORT.pdf) and the [implementation notes](code/docs/STAGE143_IMPLEMENTATION_EXPLAINER_20260926.md).

## Reproduce the score

Use Python 3.12. From `code/`, install PyTorch for your machine, then the pinned dependencies in `requirements.txt` and `requirements_stage143.txt`. The detailed installation instructions are in [code/README.md](code/README.md). Extract the matching checkpoint ZIP over this repository so that the checkpoint and ONNX graph land under `code/`.

```bash
cd code
python evaluate.py \
  --checkpoint checkpoints/stage143-openvino-order6.pt \
  --device cpu --precision fp32 --threads 4 \
  --split test --output reproduced-test.json
```

The ZIP's `BUNDLE_MANIFEST.json` lists the release commit and hashes for all inference files. The local Windows four-thread resource check measured 3.62 times baseline CPU time, 2.18 GB peak RSS and 55.81 MB of inference assets. A separate Linux one-thread check took 5.50 times its baseline, over the course's 5-times limit on that host. Timing must be checked again on the review machine; the Windows pass is not a universal guarantee.

## What the experiments show

At equal numbers of training targets, R-Drop lowered complete validation BPB from 1.464994 to 1.450613. Changing the backbone to the attention/convolution model lowered it further to 1.428594, although that comparison also changes model width and does not isolate convolution. Adding a fixed-weight training-derived count model lowered the latter to 1.423206. Later training, distillation and a validation-selected gate reached 1.399686; those later steps are not one equal-budget ablation.

The repository also records unsuccessful experiments. A two-model combination gave lower validation BPB but exceeded the inference budget, so it was not submitted. Full protocols, results and stopped trials are kept in the [experiment history](EXPERIMENT_HISTORY_20260929.md) and `code/results/`. Validation was used repeatedly, so its best score may be optimistic. No alternative was selected using the test split.

## Reuse and AI assistance

The course supplied the baseline, data, tokenizer and evaluation pipeline. The implementation draws on published work on RoPE, RMSNorm, SwiGLU, R-Drop, modified Kneser--Ney smoothing and prefix copying; the report cites the relevant papers. OpenAI Codex substantially assisted with experiment planning, code, debugging, execution, analysis and writing. [The detailed record](code/docs/AI_ASSISTANCE.md) distinguishes those contributions from the supplied code and published ideas. I remain responsible for understanding and checking the submitted implementation and results.
