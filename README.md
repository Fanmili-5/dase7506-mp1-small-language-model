# DASE7506 Project 1: small language model

A language model trained from scratch on the supplied WikiText-2 training split. The data, BPE-2048 tokenizer and evaluator are unchanged.

| Predictor | Validation BPB | Test BPB |
| --- | ---: | ---: |
| Course baseline | 2.072081 | 2.102015 |
| Final hybrid model | 1.399686 | 1.415658 |

The final model combines an eight-block attention/convolution network, within-window copying, and a training-derived six-gram modified Kneser–Ney model. A confidence gate mixes their distributions. OpenVINO runs the neural feature graph on CPU; it does not change the prediction rule.

## Read and reproduce

- [Report](REPORT.pdf): model, controlled comparisons, results and limitations.
- [Installation and evaluation](code/README.md): reproduce the frozen score without training.
- [Training recipe](TRAINING.md): neural lineage, count construction and gate fitting.
- [Results and evidence](RESULTS.md): experiment table and original measurement records.
- [Assignment](GUIDE.md): supplied rules.

Download the checkpoint ZIP from the [coursework release](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/releases/tag/mp1-coursework-20260929) and use the code snapshot attached to that same release. Extract the checkpoint ZIP at the repository root. Its manifest binds the code commit, report and all inference assets.

After installing dependencies, run from `code/`:

```bash
python scripts/verify_submission.py
python evaluate.py --checkpoint checkpoints/stage143-openvino-order6.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

The recorded complete-test result is **1.415657616535609 BPB**. “Stage143” in filenames identifies this frozen predictor. New training runs do not overwrite it.

## Resource limits

The Windows four-thread comparison measured 3.62× baseline CPU scoring time, 2.18 GB peak RSS and 55.81 MB of inference assets. A separate Linux one-thread comparison measured 5.50× baseline time, exceeding the 5× limit on that host. Timing must be verified on the review machine; the Windows result is not a universal pass.

## Reuse and AI assistance

The baseline, data, tokenizer and evaluator were supplied by the course. Published methods and data attribution are listed in the report and code README. OpenAI Codex substantially assisted with implementation, experiments, debugging, analysis and report writing. I am responsible for understanding and checking the submission.

The default branch contains the final coursework and its supporting evidence. The [original development history](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/tree/671261f374bd54b28fe6ec3ce6ab890c0b6c72c8) is retained separately; it is not needed to run the submission.
