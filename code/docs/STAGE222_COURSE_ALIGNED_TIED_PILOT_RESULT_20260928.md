# Stage222 result: shared-depth architecture rejected

Stage221 failed its original 1.25x feature-time preflight (measured
1.277846x) and remains stopped. Stage222 separately admitted the exact
unchanged architecture for a validation-only quality pilot using the
course-aligned projected total CPU ratio of **4.293104x** against its
predeclared 4.5x pilot ceiling. This estimate is **not** an actual complete
resource qualification.

The Windows RTX 3070 Ti pilot completed all fixed 2,400 seed-17 steps with
19,660,800 primary next-token targets, 688.615 s of training and 19.208 s
of complete-validation scoring. Its fixed endpoint was **1.5208217113 BPB**
over the full validation split. The same-step Stage54 control was
**1.5199503686 BPB**, so the candidate was **0.0008713427 worse**, rather
than at least 0.030 better. Intermediate evaluations were diagnostic only;
none was selected. The run wrote the fixed checkpoint SHA-256
`0cc97aa4286255b0302c0e1098b0addd414e0922c11dca7a700be1518cd2e87a`
on Windows, but this rejected model is not a release candidate.

Decision: stop this architecture. Do not run the full 7,200-step schedule,
build final inference assets, score test or replace Stage143. Evidence is
[`../results/stage221-evidence/preflight.json`](../results/stage221-evidence/preflight.json),
[`../results/stage222-course-aligned-tied-run.json`](../results/stage222-course-aligned-tied-run.json),
[`../results/stage222-course-aligned-tied-metrics.json`](../results/stage222-course-aligned-tied-metrics.json),
and [`../results/stage222-course-aligned-tied-status.json`](../results/stage222-course-aligned-tied-status.json).

The protected Stage143 validation BPB remains 1.399686162 and its already
frozen complete-test BPB remains 1.415657617. Neither the student's 1.35
goal nor 1.38 minimum has been met.
