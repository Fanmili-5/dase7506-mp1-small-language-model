# DASE7506 Project 1: small language model

I trained a small language model from scratch on the supplied WikiText-2 training split. I kept the data, BPE-2048 tokenizer and evaluator unchanged.

| Predictor | Validation BPB | Test BPB |
| --- | ---: | ---: |
| Course baseline | 2.072081 | 2.102015 |
| Hybrid model | 1.399686 | 1.415658 |

My model combines an eight-block attention/convolution network, within-window copying, and a six-gram modified Kneser–Ney model built from training text. A confidence gate mixes their distributions. I use OpenVINO to run the neural feature graph in FP32, in blocks of eight independent windows.

## Read and reproduce

- [Report](REPORT.pdf): model, controlled comparisons, results and limitations.
- [Installation and evaluation](code/README.md): run the model without retraining.
- [Training procedure](TRAINING.md): schedules, count construction and gate fitting.
- [Results](RESULTS.md): comparisons, training cost and CPU measurements.
- [Assignment](GUIDE.md): supplied rules.

Download the code and checkpoint ZIPs from [v1.0](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/releases/tag/v1.0). Extract the code into an empty directory, then extract the checkpoint into the same directory.

After installing dependencies, run from `code/`:

```bash
python scripts/verify_submission.py
python evaluate.py --checkpoint checkpoints/final-model.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

The complete-test result is **1.4156576308174311 BPB**, recorded in [test.json](code/results/final-evidence/test.json). I used validation for model selection; the leaderboard uses the test score.

## Resource limits

On the Linux four-thread rerun, median scoring time was 51.019 seconds versus 10.988 seconds for the baseline: 4.643×, with 2.14 GiB peak RSS and 55.81 MB of inference assets. I measured each predictor three times in fresh processes. An earlier run of the same files measured 5.360× on another runner, so relative speed depends on the machine. Both measurements are in [RESULTS.md](RESULTS.md).

## Reuse and AI assistance

The course supplied the baseline, data, tokenizer and evaluator. I cite the methods in the report and retain the data attribution in the code README. I used OpenAI Codex for substantial assistance with coding, experiments and writing. I am responsible for the submitted work.
