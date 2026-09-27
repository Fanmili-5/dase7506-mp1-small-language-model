# Stage143 acceptance and resource triage - 28 September 2026

This is a status and feasibility audit, not a new predictor, validation score,
test run, or permission to tune on test. The user's desired score is below
1.35 BPB, with 1.38 BPB as the minimum acceptable outcome. The frozen
Stage143 complete CPU FP32 test result is **1.415657616535609 BPB** over
428,405 targets and 1,292,013 raw bytes. It misses 1.38 by 0.035657617 BPB
and 1.35 by 0.065657617 BPB. The corresponding development-validation score
was 1.399686162042141 BPB. Stage143 is reproducibly measured but **does not
meet the student's acceptance criterion**; the existence of a report and ZIP
must not be described as achieving that criterion.

## Previously measured alternatives, without new test access

The strongest recorded complementary predictor is the fixed Stage143/Stage155
50:50 mixture at 1.376182664 validation BPB. That number is a diagnostic,
not a deployable or test-scored model. Even before adding Stage155's output
head, its 47,960,933-byte feature graph plus the 55,810,412-byte Stage143
inference set would total **103,771,345 bytes**, exceeding the 64 MiB
(67,108,864-byte) cap by **36,662,481 bytes**. Compression alone would not
establish CPU compliance.

On the same Windows machine, the random-weight Stage155 graph required a
1.8849465-second median per 32-window feature call versus 1.2561308 seconds
for Stage143's graph. Complete validation contains 1,472 windows, or 46
32-window calls. Adding 46 Stage155 calls to Stage143's measured 86.0511059
seconds projects about **172.76 seconds**, or **7.26x** the measured
23.7861255-second baseline. This is a screen, not an official resource
measurement of a trained mixture; optimization or sharing could change it.
The large margin above 5x, together with the asset overage, rules out simply
packaging the independent two-model mixture as a submission candidate.

Previous single-backbone alternatives do not close the quality gap: Stage155
averaged neural scored 1.409877270 validation BPB; its train-only
distillation into Stage143 capacity improved by only 0.000246 BPB. Shared
branch Stage154's matched early gain was 0.005998 BPB and Stage186's was
0.005196 BPB, both below their preregistered continuation gates. Richer
validation-fitted expert gating failed its cross-half transfer check. These
observations reject the *measured variants*, not all future architectures.

## 28 September evidence addendum

After this audit, two additional non-test routes were completed and stopped.
Stage198's full-vocabulary, token-conditioned count/neural gate scored
1.398856413 BPB in fixed out-of-half validation, only 0.000829749 better
than Stage143 and far above its 1.365 screening gate. The fitted validation
offsets are not deployable. Stage199 reassessed the previously untrained
full-width parallel attention/convolution architecture: the original 1.25x
feature-time screen was stricter than the course's 5x complete-predictor
budget, and a new projection suggested the architecture might fit. Its
fixed seed-17, 2,400-step pilot nevertheless gained only 0.005609248 BPB
against the same-target Stage54 control, below its 0.030 full-run gate.
No compact export or formal CPU qualification followed. Details and raw
records are in the Stage198/199 result documents and result JSON files.

The accepted Stage143 neural lineage already presented at least 255,225,110
primary training targets. Its Stage109 teacher-only, Stage120 stronger
hard-label and Stage125 mixture-aware continuations all regressed on complete
validation. Together with the Stage198/199 results, this does not prove a
sub-1.35 model impossible, but it rules out claiming that another nearby
continuation, scalar gate or parallel-width sweep is an evidence-backed
route to the student's threshold. A new experiment should start only from a
distinct mechanism with an explicit quality/resource gate, not a seed or
minor hyperparameter change.

## Boundary and next decision

The assignment requires training from supplied training text, validation for
development/model selection, an unchanged evaluator and a method freeze before
test. Stage143 has already been frozen and test-scored. Its test result must
not be used as a tuning target, a mixture weight selector, or a reason to
rerun nearby variants. Further train/validation-only research can be kept
separate from this frozen submission record, but whether a newly developed
candidate may replace an already test-scored one for the course submission
requires clarification from course staff. No second test run or course-site
submission is authorized by this audit.

Primary evidence: `../results/stage143-evidence/final.json`,
`../results/stage143-evidence/freeze-stage143-20260927.json`,
`../results/stage143-evidence/test-stage143-20260927.json`,
`../results/stage156-evidence/complementarity.json`,
`../results/stage155-evidence/feature-preflight.json`, and the Stage154,
Stage169, Stage176, and Stage186 plans/results under `code/docs/`.
