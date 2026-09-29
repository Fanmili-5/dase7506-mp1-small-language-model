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

## Initial static candidate: complete validation measurement

The candidate executes the original hash-pinned FP32 feature graph in blocks of
eight independent windows. It retains the original heads, count tables and gate.
Export checks establish identical checkpoint tensors and configuration; the
checkpoint is reserialized only to select the candidate implementation.
These measurements apply to the execution source at commit `4655b11`.

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

The first four-thread Linux comparison at commit `403bd83` also exceeded the
limit: baseline median 7.159222 s, candidate median 37.217258 s, ratio 5.198506,
peak RSS 2,295,668,736 bytes. Both processes had four available logical CPUs and
requested four PyTorch threads, but OpenVINO selected only two feature workers.
The three candidate validation scores were all 1.3996861775854912.
[Complete measurement](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36564423243).

A follow-up at commit `0d335dc` explicitly permits OpenVINO logical workers
while retaining the same four-thread ceiling. On that Linux runner, the baseline
used four PyTorch threads and the candidate used four PyTorch/OpenVINO threads.
Three complete validation repetitions measured baseline median 5.280414 s,
candidate median 28.303853 s, ratio **5.360158**, and peak RSS 2,292,768,768 bytes.
The validation BPB remained 1.3996861775854912. The timing limit still fails on
that host. Do not compare absolute seconds across the two CI hosts as a speedup.
[Logical-worker measurement, attempt 1](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36565249601/attempts/1).

This changes scheduling only, not weights or precision. The Windows run also
passed all 57 tests without skips and the synthetic numerical/causality checks,
with four actual OpenVINO workers. Three complete validation repetitions gave
baseline median 22.910182 s, candidate median 74.373097 s, ratio **3.246290**,
and peak RSS 2,207,744,000 bytes. All three validation scores were
1.399686161908931. The raw reports are `resources-windows-logical4.json` and
`equivalence-windows-logical4.json` in the independent Windows audit checkout's
`code/results/runtime-audit/` directory, and were retrieved locally. The exact
candidate implementation SHA-256 is
`111e34302cf1095a18c57fc4d01f880ce1cc11e58316eac381d8ba38776c8c12`.
None of these candidate records replaces the released
model's Windows qualification. No rule requiring Linux single-thread execution
was found in the supplied course documents; the Linux measurements document
cross-host risk rather than a separate single-thread submission requirement.

## Same-commit Linux recheck

At the user's request, run `36565249601` was rerun once at the identical commit
`0d335dce93cfed69ded78bf8d1f44efa1899e387`. This was a repeat of CPU scoring,
not training. The method and weights were unchanged; neither run scored test.
Both attempts used the complete validation split, CPU FP32, four requested
threads, four actual OpenVINO workers and three alternating fresh-process
repetitions for each predictor.

| Linux attempt | Baseline seconds, repetitions 1 / 2 / 3 | Candidate seconds, repetitions 1 / 2 / 3 | Ratio of medians | Time limit |
| --- | --- | --- | ---: | --- |
| 1 | 5.280414 / 5.261645 / 5.330604 | 28.525318 / 28.303853 / 28.225612 | 5.360158 | Exceeded |
| 2 | 11.242709 / 10.987902 / 10.973218 | 51.150770 / 51.018586 / 50.893668 | 4.643160 | Met on this runner |

Attempt 2 finished on 29 September 2026 at 12:21 UTC. Its baseline median was
10.987902 s and candidate median was 51.018586 s. Peak candidate RSS was
2,295,070,720 bytes, below 4 GiB. All three candidate validation scores were
1.3996861637466336. The suite reported 57 tests, with the two CUDA-only tests
skipped on this CPU runner; synthetic equivalence and causality checks passed.
[Attempt 2 logs](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36565249601/attempts/2)
and [raw JSON artifact](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36565249601/artifacts/11032886588)
retain the new evidence.

Both attempts have identical baseline/candidate checkpoint, implementation,
evaluator and tokenizer hashes. In particular, the candidate checkpoint SHA-256
is `b8c1273f9ae629f7e6667370177ff4312ecf9aead5209abd4319a3bb62aa67a4`.
The runtime environment was not fully identical: both attempts exposed four
logical CPUs, but the default PyTorch inter-op count was two in attempt 1 and
four in attempt 2, matching between baseline and candidate within each attempt.
CPU model and host contention were not recorded, so these measurements do not
identify the precise cause of the timing difference.

Thus 5.360158 is a valid observation, not a universal Linux multiplier, and
4.643160 is a valid pass on the second runner, not a guarantee for every host.
Both outcomes must be retained. Do not pool seconds across hosts, call the
rerun an algorithmic speedup, or rerun repeatedly just to select a passing host.
This recheck does not replace or modify the published coursework bundle.
