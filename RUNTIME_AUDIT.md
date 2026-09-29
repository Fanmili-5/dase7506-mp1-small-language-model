# CPU runtime audit, 29 September 2026

This branch is an execution-only investigation, not the coursework release.
The default submission branch and the published model bundle remain unchanged.
No model was trained and no test score was selected during this audit.

## Thread-count interpretation

The original course `evaluate.py` defaults to four threads, and the supplied
README describes a four-thread reference CPU. Neither supplied document imposes
a separate Linux-only, single-thread requirement. One-thread Linux timing is an
additional portability check, not a mandated submission configuration. Its
failure must not be presented as proof that all permitted configurations fail.
Fair timing comparisons use the same machine, CPU FP32 scorer, workload and
requested thread budget for both predictors. Hardware availability and any
runtime reduction in actual worker count should be recorded explicitly.

## Complete validation measurement

The candidate executes the original hash-pinned FP32 feature graph in blocks of
eight independent windows. It retains the original heads, count tables and gate.
Export checks establish identical checkpoint tensors and configuration; the
checkpoint is reserialized only to select the candidate implementation.

| Host | Threads | Repetitions | Candidate / baseline median time | Peak RSS, bytes | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Windows 11, PyTorch 2.7.1+cu126 | 4 | 3 | 3.252458 | 2,208,567,296 | Within measured limits |
| Linux x86-64, PyTorch 2.7.1+cpu | 1 | 3 | 5.268078 | 2,296,111,104 | Fails the 5x time limit |

Both comparisons use the complete validation split and fresh processes with
alternating baseline/candidate order. Windows candidate BPB was
1.399686161908931 and Linux candidate BPB was 1.399686163746634.
The Linux run and its raw resource/equivalence JSON are attached to
[run 36560764563](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36560764563).
Windows raw JSON was retrieved from the independent remote audit checkout.
Do not compare absolute seconds across separate CI jobs as a paired speedup.

## Follow-up diagnostics

These small paired probes use validation inputs but do not score targets. They
are not substitutes for the complete resource measurement above.

- Splitting the entire predictor into eight-row blocks, including heads and
  counts, left outputs exactly unchanged. It saved only about 1--2% on the
  measured Linux host and was slightly slower on Windows. Not adopted.
- Four-row and one-row feature blocks did not improve on eight rows on either
  host. [Linux evidence](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36562998173).
- Original PyTorch kernels were tested after recovering every neural parameter
  exactly from the frozen graph/checkpoint, including the tied embedding. They
  were slower than static OpenVINO on both hosts. Maximum hidden-state
  differences were below 8e-6. Not adopted.
  [Linux evidence](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36563669660).
- A Windows diagnostic initially stopped because it read UTF-8 text using the
  system GBK default. Explicit UTF-8 handling fixed it; both Windows probes then
  completed. This did not affect the course evaluator or the released model.

The candidate fails the additional one-thread Linux check. A four-thread Linux
comparison is being checked separately using the course evaluator's default
budget. Do not replace the frozen release or reuse its qualification records
for this branch without a completed audit of the exact candidate.
