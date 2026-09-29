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
- [Results and evidence](RESULTS.md): experiment table and original measurement records.
- [Assignment](GUIDE.md): supplied rules.

Download the checkpoint ZIP from the [final release](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/releases/tag/mp1-final-20260929) and use the code snapshot attached to the same release. Extract the checkpoint ZIP at the repository root. Its manifest identifies the code commit, report and inference files.

After installing dependencies, run from `code/`:

```bash
python scripts/verify_submission.py
python evaluate.py --checkpoint checkpoints/final-model.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

The complete-test result is **1.4156576308174311 BPB**, recorded in [test.json](code/results/final-evidence/test.json). This is the submission score; validation BPB is only for model selection. Stage numbers in supporting filenames identify earlier experiments.

## Resource limits

On the recorded Linux four-thread rerun, median scoring time was 51.019 seconds versus 10.988 seconds for the baseline: **4.643×**, with **2.14 GiB** peak RSS and **55.81 MB** of inference assets. Each predictor was measured three times in fresh processes. An earlier run of the same files measured 5.360× on another runner, so CPU timing remains machine-dependent. [RESULTS.md](RESULTS.md) retains both measurements.

## Reuse and AI assistance

The course supplied the baseline, data, tokenizer and evaluator. I cite the methods in the report and retain the data attribution in the code README. I used OpenAI Codex for substantial assistance with coding, experiments and writing. I am responsible for the submitted work.

The default branch contains the final coursework and its supporting evidence. The [original development history](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/tree/671261f374bd54b28fe6ec3ce6ab890c0b6c72c8) is retained separately; it is not needed to run the submission.
