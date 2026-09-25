# MP1 candidate-selection audit — 26 September 2026

This is a validation-only decision record, **not** a method freeze, test score,
or submission. Its purpose is to keep exploratory lower numbers distinct from
predictors that have actually passed the course's CPU FP32, memory, and asset
checks. All BPBs below use the complete 376,599-target validation split unless
marked as a diagnostic bound. The fixed target is below 1.35 BPB, but a target
is not evidence that any current predictor attains it.

| Candidate or diagnostic | Validation BPB | What the number establishes | Submission status |
| --- | ---: | --- | --- |
| Stage143 compact single-pass Transformer + copy + order-six count expert | **1.399686162** | Complete Windows CPU FP32 validation; 3.617702× baseline CPU time, 2,176,729,088-byte peak RSS, 55,810,412-byte conservative inference assets | **Best current resource-qualified development candidate**; not frozen or test-scored |
| Stage85 earlier qualified hybrid | 1.403024133 | Complete validation and 4.900108× CPU, 2,041,077,760-byte RSS, 48,570,230-byte assets | Qualified but worse than Stage143 |
| Stage155 larger pure neural model | 1.409877270 | Complete validation of one expert | No compact export/resource qualification |
| Stage143/Stage155 fixed 50:50 mixture | 1.376182664 | Demonstrates complementarity between two separately stored predictors | Diagnostic only; over 64 MiB and no whole-predictor resource qualification |
| Stage169 teacher-trained Stage143-capacity student | 1.399440139 | Matched 900-step endpoint, 0.000246023 better than Stage143 | Failed predeclared 0.005 continuation gate; no compact export/resource qualification |
| Stage183 geometric fusion, beta 0.5 | 1.398247514 | Normalized full-distribution diagnostic, 0.001438648 better than Stage143 | Failed predeclared 0.015 advancement gate; no deployed inference/resource qualification |
| Stage176 causal-gate cross-fit | 1.400044877 | Out-of-half validation-fitted diagnostic | Worse than Stage143; validation-fitted coefficients cannot be deployed |
| Stage175 answer-aware expert oracle | 1.300186476 | Unattainable lower bound for a per-target choice of the frozen experts | Uses the correct target to choose an expert; illegal at inference |
| Stage188 same-support Witten–Bell count swap | 1.414884002 | Complete Windows CPU FP32 validation of a train-only, frozen-gate alternative | Regressed; no follow-on resource qualification |

The primary records are [Stage143 qualification](../results/stage143-evidence/final.json),
[Stage85 qualification](../results/stage85-evidence/final.json),
[Stage156 complementarity](../results/stage156-evidence/complementarity.json),
[Stage169 pilot](../results/stage169-evidence/metrics.json),
[Stage175 oracle](../results/stage175-evidence/current-expert-oracle.json),
[Stage176 cross-fit](../results/stage176-evidence/causal-gate-crossfit.json),
[Stage183 fusion](../results/stage183-expert-fusion.json), and
[Stage188 validation](../results/stage188-evidence/validation.json).
The Stage155 standalone score is recorded in Stage156's complementarity
result. This table is an audit of the listed leading alternatives, **not** an
exhaustive proof that no untested architecture can beat Stage143.

## Decision and unresolved risk

Do not promote an oracle, nondeployable mixture, validation-fitted rule, or
failed-gate diagnostic because its numerical BPB is lower. The only current
candidate eligible for a deliberate method freeze is Stage143. It remains
0.049686162 BPB above the aspirational 1.35 line. The Windows resource pass
does not prove host-independent portability: a separate Linux one-thread
measurement was 5.502555× baseline, and the two-vCPU CI host cannot run a
four-thread equivalent. The [Stage187 profile](STAGE187_LINUX_NODE_PROFILE_20260926.md)
found dense FP32 projections dominate that runtime, without establishing an
equivalent rewrite that closes the gap.

The [assignment guide](../../GUIDE.md) requires a student ID and full-test
BPB on the course site **before 29 September 2026**, with immutable code and
matching checkpoint links plus a <=10-page report by 30 September (UTC+8).
Stage143 has **not** been method-frozen or test-scored; the historical
`REPORT.pdf` still describes an older candidate. A student decision on the
freeze timing is needed before the one-time full-test; the student ID is needed
for the course-site submission. Until then, preserve Stage143's committed
hashes and do not use test data for selection.
