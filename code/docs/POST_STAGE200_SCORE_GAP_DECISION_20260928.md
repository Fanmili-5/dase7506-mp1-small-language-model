# Post-Stage200 score-gap decision (28 September 2026)

This is a decision audit, not a new predictor, validation outcome, freeze, or
test run. The student's ideal objective remains complete-validation BPB
strictly below 1.35; their minimum acceptable complete-test score is below
1.38. The assignment itself ranks the lowest reproducible full-test BPB
under its fixed data, evaluation and resource rules. Neither student
threshold has been reached.

## Measured position and scale of the gap

- The protected and already frozen Stage143 predictor has complete-validation
  BPB `1.399686162042141` and complete CPU-FP32 test BPB
  `1.415657616535609`. The validation gap to 1.35 is `0.049686162` BPB;
  the tested gap to 1.38 is `0.035657617` BPB. The latter is an acceptance
  status, **not** a target used to tune future methods.
- Stage143 passed the Windows course-style CPU/RAM/asset checks at
  `3.617702x`, `2,176,729,088` bytes and `55,810,412` bytes respectively.
  Against 64 MiB, the remaining uncompressed asset margin is `11,298,452`
  bytes. Independent Linux one-thread timing was `5.502555x`, so Windows
  qualification is not a universal CPU guarantee.
- The best measured complementary neural mixture, Stage143/Stage155 50:50,
  reached `1.376182664` **validation** BPB but was not a deployable model:
  independent assets would exceed 64 MiB by at least `36,662,481` bytes and
  its Windows complete-runtime projection was `7.26x` baseline. It also
  remains `0.026182664` above the student's 1.35 validation objective.

## Why the nearby routes do not justify another blind run

| Distinct mechanism | Complete development evidence | Next action |
| --- | --- | --- |
| Stage197 shared-trunk heterogeneous branches | Selected complete validation `1.4268634`, worse than Stage143 | Stop this trained architecture |
| Stage199 parallel attention/convolution | Matched 2,400-step gain `0.005609248` vs required `0.030` | Stop before full training |
| Stage195/196 frozen-feature residual experts | Best gain about `0.0012` from Stage143 | Do not stack residuals or retune weights |
| Stage198 token-conditioned neural/count gate | Out-of-half complete validation `1.398856413`, gain `0.000829749` | Do not deploy validation-fitted offsets |
| Stage200 train-only adaptive target margin | Matched 2,400-step gain `0.001180297` vs required `0.030` | Stop; no alpha/seed search |
| Stage176 rich causal gate between current experts | Out-of-half `1.400044877`, worse than Stage143 | Do not train another nearby gate |

The entries are separate experiments with different controls; they are not
pooled into an expected effect size. They do show that no **measured**
incremental route closes a material fraction of the remaining gap. The
Stage175 `1.300186476` target-aware oracle is explicitly illegal at inference
and cannot be used as a projected score. A fixed train-only word-prefix
expert and same-support Witten-Bell substitution were also too weak or
regressive. A new candidate would need a distinct *core predictor* mechanism
and an explicit same-target quality/resource gate; small post-hoc validation
grids are not justified by this evidence.

## 28 September Stage202 addendum

One distinct core-layout hypothesis was then predeclared and tested:
reordering the unchanged four causal-convolution/four attention blocks from
alternating to local-first/global-later. Its same-seed, same-target 2,400-step
complete validation endpoint was `1.521048958`, worse than the Stage54
`1.519950369` control by `0.001098590`. The fixed +0.030 BPB continuation
gate failed, so no full training, CPU predictor qualification or test followed.
This rejects this ordering, not every future core architecture. See the
Stage202 plan, result and raw evidence; Stage143 remains protected.

## Deadline-aware release boundary

Keep Stage143's frozen checkpoint, source, graph and test record protected.
Do not describe the prepared ZIP/report as meeting 1.38 or 1.35. The bundle
is a lawful fallback artifact only, not a score success. The local guide asks
for an initial student-ID/score issue **before 29 September (UTC+8)** and
matching immutable code/checkpoint links by 30 September; the live course
site currently displays a 30 September score deadline. Use the earlier
instruction conservatively until staff clarify the discrepancy. Publishing
the student's ID/score or making the private repository public requires the
student's explicit choice; no such choice is recorded here.

Authoritative evidence: Stage143 final/freeze/test/resource JSON, Stage156
complementarity and Stage175 oracle JSON, and Stage176/195/196/197/198/199/200
plan/result documents and raw JSON under `code/results/`. This audit is not
permission to run another full test or to submit.
