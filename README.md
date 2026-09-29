# DASE7506 Project 1: small language model

I trained a small language model from scratch on the supplied WikiText-2 training split. I kept the data, BPE-2048 tokenizer and evaluator unchanged.

| Predictor | Validation BPB | Test BPB |
| --- | ---: | ---: |
| Course baseline | 2.072081 | 2.102015 |
| Final hybrid model | 1.399686 | 1.415658 |

My model combines an eight-block attention/convolution network, within-window copying, and a six-gram modified Kneser–Ney model built from training text. A confidence gate mixes their distributions. I use OpenVINO to run the neural feature graph in FP32, in blocks of eight independent windows.

## Read and reproduce

- [Report](REPORT.pdf): model, controlled comparisons, results and limitations.
- [Installation and evaluation](code/README.md): reproduce the frozen score without training.
- [Training recipe](TRAINING.md): neural lineage, count construction and gate fitting.
- [Results](RESULTS.md): required comparisons, training cost and final measurements.
- [Assignment](GUIDE.md): supplied rules.

Download the code and checkpoint ZIPs from the [submission release](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/releases/tag/mp1-submission-20260929). Extract the code ZIP into an empty directory, then extract the checkpoint ZIP into that directory. Its manifest identifies the code commit, report and inference files.

After installing dependencies, run from `code/`:

```bash
python scripts/verify_submission.py
python evaluate.py --checkpoint checkpoints/final-model.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

The complete-test result is **1.4156576308174311 BPB**, recorded in [test.json](code/results/final-evidence/test.json). This is the submission score; validation BPB is only for model selection.

## Resource limits

On the recorded Linux four-thread rerun, median scoring time was 51.019 seconds versus 10.988 seconds for the baseline: **4.643×**, with **2.14 GiB** peak RSS and **55.81 MB** of inference assets. Each predictor was measured three times in fresh processes. An earlier run of the same files measured 5.360× on another runner, so CPU timing remains machine-dependent. [RESULTS.md](RESULTS.md) retains both measurements.

## Reuse and AI assistance

The course supplied the baseline, data, tokenizer and evaluator. I cite the methods in the report and retain the data attribution in the code README. I used OpenAI Codex for substantial assistance with coding, experiments and writing. I am responsible for the submitted work.

The default branch contains the submitted model, training and evaluation code, tests, report and supporting measurements. Some training modules retain numbered filenames because later steps import them. Older checkpoints and search logs are kept in the [development history](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/tree/671261f374bd54b28fe6ec3ce6ab890c0b6c72c8), outside the submission package.
