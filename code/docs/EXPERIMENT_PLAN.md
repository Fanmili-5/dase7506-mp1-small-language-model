# Experiment plan

## Objective and non-negotiable rules

The objective is lower validation and frozen-test bits per byte (BPB), not token
perplexity. The data, tokenizer, independent 256-token causal windows, and fixed
evaluator remain unchanged. No model choice, interpolation weight, or checkpoint
choice may use test results.

The supplied baseline processes 9,830,400 training targets:

```text
1200 optimizer updates × 32 sequences × 256 targets
```

Formal equal-budget comparisons use this same target count. When gradient
accumulation is enabled, processed targets are calculated as:

```text
updates × micro_batch_size × grad_accum × 256
```

## Candidate mechanisms

The initial framework exposes mechanisms separately so results remain
interpretable:

| Config | Difference from the matched control | Status |
|---|---|---|
| `student_control.json` | Reimplementation of supplied block | correctness control |
| `student_rope.json` | learned positions → RoPE only | clean primary candidate |
| `student_swiglu.json` | GELU FFN → parameter-matched SwiGLU only | clean primary candidate |
| `student_modern.json` | RoPE + RMSNorm + SwiGLU + bias removal + scaled residual init | exploratory bundle |
| `student_scaled.json` | larger modern model | later capacity experiment |

These are hypotheses until validation experiments are complete. The final report
must not describe an untested mechanism as an improvement.

## Stage 0 — local framework validation

1. Verify fixed-file hashes.
2. Run all contract and configuration tests.
3. Run a two-update CPU smoke test.
4. Confirm that the smoke checkpoint reloads in the supplied evaluator.
5. Do not run test evaluation.

Exit condition: all tests pass and the run directory contains `run.json`,
`progress.json`, `metrics.json`, `checkpoint.pt`, and resumable state.

## Stage 1 — equal-target screening on Windows CUDA

Run seed 17 for:

1. official baseline;
2. RoPE-only;
3. SwiGLU-only;
4. exploratory modern bundle.

All runs use 1,200 updates, effective batch 32, baseline optimizer and learning
rate schedule, and exactly 9,830,400 processed targets. Validation checkpoints at
steps 300, 600, 900, and 1,200 are diagnostics; the final comparison is at the
same update count.

Primary decision metric: validation BPB. Secondary metrics: training stability,
training seconds, parameter count, CPU FP32 validation time, and checkpoint size.

## Stage 2 — isolate and replicate

Select at most one clean mechanism based on Stage 1 validation. Run the selected
mechanism and its matched ablation with two additional predeclared seeds. Report
all seeds; do not select only the lucky one.

If the modern bundle wins but neither clean mechanism wins, split the bundle into
additional one-factor ablations before making a causal claim.

### Stage 2 preregistration — 2026-09-19

Before running new models, fix the additional seeds to 23 and 42. Run baseline,
RoPE-only, and modern bundle at both seeds: six runs. Combine with the existing
seed-17 runs and report all three paired results and mean/sample standard
deviation; do not report only the best seed. Each run uses 1,200 updates, batch
32, the original optimizer/schedule, and 9,830,400 processed targets.

At seed 17, run three additional configurations to complete this component
ladder (the endpoints already exist):

1. RoPE-only (existing).
2. RoPE + approximately parameter-matched SwiGLU.
3. Add RMSNorm.
4. Remove linear biases.
5. Add scaled residual initialization (existing modern bundle).

This ladder estimates conditional differences in the stated order, not global
independent component effects. Its intermediate configurations have only one
seed, so small differences need replication before claims or final selection.
All nine new runs train from scratch; the original seed-17 checkpoints are
reused only as comparison evidence, not as initialization. Total new training
budget is 88,473,600 targets. Score final equal-target endpoints with the fixed
CPU FP32 validation evaluator. No test calls.

## Stage 3 — capacity and training-budget search

### Stage 3A preregistration — 2026-09-19

Before observing any stage-3 result, run five fresh-initialization experiments:

1. RoPE+SwiGLU, seeds 23 and 42, at 1,200 updates to complete the same three-seed
   replication used for baseline, RoPE-only, and modern.
2. Baseline, modern, and RoPE+SwiGLU, seed 17, at 4,800 updates. All three use
   effective batch 32, the original optimizer hyperparameters, and one cosine
   schedule planned for 4,800 updates from the start. Each processes 39,321,600
   targets; old 1,200-update checkpoints are not resumed.

The long runs record full GPU FP32 validation at steps 600, 1200, 1800, 2400,
3000, 3600, 4200, and 4800; their final equal-target endpoints are re-scored on
CPU FP32 validation. The two replication runs use the existing 300-step cadence.
Total new budget is 137,625,600 targets. No stage-3 choice uses test.

Decision gate: compare every predeclared run, retain the simpler candidate only
if its three-seed evidence is competitive, and use the long-run curves to decide
whether the next bottleneck is capacity, schedule, or overfitting. Do not compare
the 1,200-step short-cosine endpoint to a 1,200-step point inside the 4,800-step
cosine schedule as if they used the same training recipe.

### Stage 3A result — 2026-09-19

All five preregistered runs completed and were re-scored on complete CPU FP32
validation. RoPE+SwiGLU's three-seed 1,200-step result is 1.837202 mean BPB with
0.004797 sample standard deviation. At 4,800 steps and 39,321,600 targets, the
seed-17 endpoints are 1.754263 for baseline, 1.698374 for modern, and 1.700166
for RoPE+SwiGLU. All three curves still improve at the end, though the final
600-step gain is small. Modern leads the simplified model by only 0.001791 BPB;
this is a direct single-seed result rather than evidence of robust superiority.

Decision: carry modern forward as the primary capacity-search mechanism because
it is marginally better in the equal-budget long run and has marginally fewer
parameters. Retain RoPE+SwiGLU as the simpler fallback. Screen the existing
approximately three-million-parameter modern configuration for 1,200 steps,
then run the fixed CPU FP32 time/RAM gate before authorizing its long run. This
gate uses validation only and does not change the Stage 1–2 mechanism evidence.

### Stage 4A preregistration — 2026-09-19

Train `student_scaled.json` from random initialization with seed 17 for 1,200
updates, effective batch 32, the original optimizer, and the same short cosine
schedule used by Stages 1–2. The 3,049,920-parameter candidate processes
9,830,400 targets. Record complete GPU FP32 validation every 300 steps and
re-score the endpoint with CPU FP32 validation.

After scoring, compare it with the Stage-3 long-baseline checkpoint using three
fresh CPU processes per model, four threads, alternating measurement order.
Require candidate median scoring time at most 5x baseline, peak RSS at most
4 GiB, and total inference assets below 64 MiB. The baseline checkpoint's
training length does not affect architecture scoring cost; it is used only as
the unchanged runtime control.

Long-run authorization requires all resource limits plus a clear short-run
quality gain over the 1,200-step modern result of 1.835656528 BPB. A marginal
or negative result triggers width/depth or optimization diagnosis instead of an
automatic 4,800-step run. No test call is allowed in this gate.

### Stage 4A result and Stage 4B authorization — 2026-09-19

The scaled endpoint reaches 1.733711899 CPU FP32 validation BPB after 1,200
steps, a 0.101944629 reduction from the same-recipe 1,200-step modern screen.
Its three fresh CPU measurements have a 30.4563-second median versus 12.2939
seconds for the baseline, a 2.4774x ratio. Maximum peak RSS is 1,955,733,504
bytes, and the checkpoint is 12,215,161 bytes. Thus the quality, 5x time, 4 GiB
RSS, and provisional 64 MiB asset gates all pass.

Authorize one fresh seed-17 4,800-step run of the same scaled configuration with
effective batch 32 and a cosine plan fixed to 4,800 steps from initialization.
It processes 39,321,600 targets, records full validation every 600 steps, and is
re-scored on CPU FP32 validation. The old 1,200-step checkpoint is evidence only
and is not resumed. Compare the endpoint directly with Stage 3's 4,800-step
modern value of 1.698374403. No test call is allowed.

### Stage 4B result — 2026-09-19

The equal-budget 4,800-step CPU FP32 endpoint is 1.660364254 BPB, improving on
the 1.05M modern control by 0.038010149 BPB. The curve reaches its minimum at
step 4,200; the saved best checkpoint is independently re-scored on CPU FP32 at
1.658827799 BPB, then the endpoint regresses by 0.001536476. Report the 4,800
endpoint for equal-budget architecture comparison and keep the 4,200 checkpoint
as the validation-selected downstream control.

Decision: more duration is not the next intervention. The 3.05M model stays
well inside the resource envelope and shows a clear capacity gain, so screen a
small number of wider/deeper configurations at the same 1,200-step recipe. Gate
the best new shape on CPU time/RAM before any long run. Keep all runs, including
failures, and do not use test.

### Stage 5A preregistration — 2026-09-19

Screen three fresh seed-17 configurations at 1,200 updates, effective batch 32,
and the same short baseline schedule as the existing 3.05M control:

1. width 192, depth 8, 3,935,424 parameters (depth intervention);
2. width 224, depth 6, 4,072,992 parameters (intermediate width);
3. width 256, depth 6, 5,247,744 parameters (near resource-envelope width).

Each run processes 9,830,400 targets and records full validation every 300
steps, followed by CPU FP32 endpoint verification. The comparison control is the
existing width-192/depth-6 result of 1.733711904 BPB. These are capacity-frontier
screens, not parameter-matched architectural ablations.

Retain every result. A candidate needs at least a clear 0.005 BPB improvement
over the control to justify a resource benchmark. Benchmark the best-quality
shape first; if it exceeds 5x CPU time or 4 GiB RSS, test the next-best shape.
Only a quality-and-resource winner may receive a fresh 4,800-step run. No test
call is allowed.

### Stage 5A result and Stage 5B resource gate — 2026-09-19

CPU FP32 1,200-step endpoints are 1.719857483 for width-192/depth-8,
1.710100868 for width-224/depth-6, and 1.690024271 for width-256/depth-6. All
three improve over the 3.05M control, and the 5.25M width-256/depth-6 candidate
leads by 0.043687633 BPB. At roughly four million parameters, the wider/shallow
shape also beats the deeper/narrow shape by 0.009756615 BPB.

Per the preregistered gate, benchmark width-256/depth-6 first against the same
Stage-3 baseline using three fresh CPU FP32 validation processes per model, four
threads, and alternating order. Authorize a long run only if median time remains
at most 5x baseline, peak RSS at most 4 GiB, and core inference assets at most
64 MiB. Otherwise resource-test the width-224/depth-6 fallback. No test call.

### Stage 5B result and Stage 5C authorization — 2026-09-19

Width-256/depth-6 has a 46.2336-second median CPU FP32 validation time versus
12.6854 seconds for baseline, a 3.6446x ratio. Its maximum peak RSS is
1,972,400,128 bytes and checkpoint size is 21,006,521 bytes. The time, memory,
and provisional asset gates therefore pass.

Authorize one fresh seed-17 4,800-step run with effective batch 32 and the same
optimizer. Fix the cosine plan to 4,800 updates from initialization; do not
resume the short run. Record complete validation every 600 steps and re-score
the endpoint with CPU FP32. Compare it with both the 3.05M scaled endpoint
1.660364254 and its validation-selected 4,200-step checkpoint 1.658827799.
Preserve the best periodic checkpoint separately from the equal-budget endpoint.
No test call is allowed.

### Stage 5C result and Stage 6A preregistration — 2026-09-19

The fresh width-256/depth-6 run reaches an equal-budget 4,800-step CPU FP32
endpoint of 1.673347560 BPB. Its validation curve reaches a minimum at step
3,000, and the independently re-scored best checkpoint is 1.656612093 BPB. This
is 0.002215706 lower than the 3.05M model's validation-selected checkpoint, but
the wide model then deteriorates by 0.016735467 BPB through step 4,800. Core
inference assets are 21,136,388 bytes; the previously measured 3.6446x CPU-time
ratio and 1,972,400,128-byte peak RSS remain inside all limits.

The gain over the smaller validation-selected control is small and single-seed,
while sustained late overfitting is clear. Before additional capacity growth,
run a matched seed-17 regularization screen with the 5.25M architecture. Train
three fresh models for 3,600 planned updates: dropout 0, 0.05, and 0.10. All use
effective batch 32, learning rate 0.001, weight decay 0.1, the baseline cosine
schedule planned for 3,600 steps from initialization, and 29,491,200 targets.
Record complete validation every 300 steps and independently score both endpoint
and validation-selected checkpoints on CPU FP32. The no-dropout run is the
matched schedule control; only the two dropout configs are regularization
interventions. Retain all results and do not use test.

Advance a recipe only if it beats the current 1.656612093 validation control
without instability. Treat improvements below 0.002 BPB as provisional and
require paired-seed replication before freezing a final model. Architecture
runtime is unchanged by training-only dropout, so the existing resource gate
continues to apply; final asset size must still be recomputed from the selected
checkpoint.

### Stage 6A result and Stage 7A preregistration — 2026-09-19

At the matched 3,600-step budget, CPU FP32 validation-selected BPB is
1.647222340 for dropout 0, 1.602405749 for dropout 0.05, and 1.597758459 for
dropout 0.10. The no-dropout optimum occurs at step 3,300, while both regularized
runs continue improving through step 3,600. Dropout 0.10 improves by 0.049463881
BPB over the matched control and by 0.058853633 over the previous Stage-5
candidate. The inference graph is unchanged in eval mode and core assets remain
21,136,388 bytes, so the Stage-5 resource gate still applies.

Before tuning duration or learning rate, extend this one-dimensional matched
screen with exactly two fresh seed-17 runs: dropout 0.15 and 0.20. Keep the same
5.25M architecture, 3,600-step cosine plan, effective batch 32, optimizer,
learning rate, weight decay, and 300-step validation cadence. Each processes
29,491,200 targets. Compare against the already completed dropout-0.10 control;
do not rerun it. Independently CPU-score endpoint and selected checkpoints and
retain both results.

If neither new setting improves on 1.597758459 by at least 0.002 BPB, retain
dropout 0.10 for duration/seed validation. If a stronger dropout wins, use the
lowest-BPB setting as the next candidate but still require replication before a
final claim. No test call is allowed.

### Stage 7A result and Stage 8A preregistration — 2026-09-19

CPU FP32 endpoint and selected BPB are 1.624384904 for dropout 0.15 and
1.646837034 for dropout 0.20. Both are worse than the existing dropout-0.10
control by 0.026626445 and 0.049078575 respectively. Retain dropout 0.10 and
stop increasing regularization strength under the 3,600-step recipe.

Because the retained curve is still improving at step 3,600, run one fresh
seed-17 duration candidate using dropout 0.10 and a cosine schedule planned for
4,800 updates from initialization. Do not resume the 3,600-step checkpoint.
Keep effective batch 32, optimizer, learning rate, weight decay, and all model
settings fixed; evaluate complete validation every 300 steps. The run processes
39,321,600 targets. Independently CPU-score endpoint and selected checkpoints.

Adopt the longer recipe only if its selected BPB improves on 1.597758459 by at
least 0.002. Otherwise retain the 3,600-step recipe. After this single duration
decision, stop seed-17 tuning and run the selected recipe at seeds 23 and 42,
reporting all three seeds. No test call is allowed.

### Stage 8A result and Stage 9A preregistration — 2026-09-19

The fresh 4,800-step seed-17 curve improves at every 300-step observation and
ends at its selected checkpoint. Independent CPU FP32 validation gives
1.575605888 BPB, an improvement of 0.022152571 over the frozen 3,600-step
dropout-0.10 control. Adopt the 4,800-step recipe and stop seed-17 tuning.

Replicate the exact frozen recipe at seeds 23 and 42: width 256, depth 6,
dropout 0.10, effective batch 32, baseline cosine schedule planned from
initialization for 4,800 updates, and validation every 300 steps. Each run
processes 39,321,600 targets. Report all three seeds and their mean and sample
standard deviation. Predeclare the prospective final checkpoint as the lowest
CPU FP32 validation checkpoint among seeds 17, 23, and 42, while explicitly
disclosing this validation-based seed selection. Before any test call, rerun the
formal CPU time/RAM/resource benchmark on that checkpoint and freeze a manifest
of source, config, checkpoint, tokenizer, and evaluator hashes. No test call is
allowed during Stage 9A.

### Stage 9A result, resource gate, and freeze — 2026-09-19

Independent CPU FP32 validation-selected BPB is 1.575605888, 1.582028099, and
1.581602162 for seeds 17, 23, and 42. The mean is 1.579745383 and the sample
standard deviation is 0.003591228. All three curves select their 4,800-step
endpoint. Apply the preregistered rule and select seed 17, explicitly disclosing
that this is validation-based selection across three seeds.

A fresh three-repeat CPU benchmark of the selected checkpoint measures a 3.485x
median-time ratio against the baseline and 1,972,150,272 bytes maximum peak RSS.
Core inference assets total 21,136,388 bytes. All three resource limits pass.
The pre-test manifest freezes source archive, implementation, config, checkpoint,
tokenizer, evaluator, training budget, and resource measurements. Only after
that manifest was written was the full test evaluated: the initial baseline is
2.102014912 BPB and the frozen candidate is 1.605136467 BPB. These test results
must not be used for further model changes, seed selection, or hyperparameter
tuning.

Only after choosing the mechanism:

1. compare width/depth options on short validation runs;
2. measure CPU FP32 scoring ratio before committing to a larger model;
3. select learning rate and training duration on validation;
4. train the final candidate;
5. verify checkpoint assets are comfortably below 64 MiB.

The equal-target evidence from Stages 1–2 remains the mechanism comparison even
if the final leaderboard model is trained longer.

### Score-first follow-up agreed on 2026-09-19

Keep the first equal-target screen small. Use its learning curves to decide
whether the next bottleneck is optimization, capacity, or overfitting; do not
assume the modern bundle wins. These are candidate interventions, not measured
improvements:

1. Compare a longer baseline schedule with the selected architecture using the
   same longer target budget. Do not extend a finished short cosine run and call
   it a fresh long-schedule comparison. Checkpoint ancestry counts toward cost.
2. Tune learning rate, schedule, weight decay, dropout, and effective batch in
   small successive groups, retaining unsuccessful runs in the search ledger.
3. Screen width/depth choices around one to several million parameters. Measure
   CPU FP32 validation scoring cost before committing to expensive training.
4. Try same-trajectory weight averaging only after a useful trajectory exists;
   choose averaging settings on validation. Independent initializations must not
   be naively averaged as though their parameters were aligned.
5. Consider a small probability ensemble or compact train-derived n-gram model
   only if a strong single model leaves enough CPU and asset budget. Include all
   inference assets and reset evaluation-prefix state at each window boundary.

Stop or revise a route when it violates the CPU budget, develops sustained
validation deterioration, or fails to improve enough to justify its complexity.
No test-based decisions are permitted. Seed-17 screening alone is not proof of
a robust improvement, and no leaderboard rank is promised.

Internal target dates: select/freeze by 27 September, evaluate and make the
initial submission by 28 September, and reserve 29–30 September for reproduction
and packaging. The local guide and live website differ in their initial-score
deadline wording; use the stricter local-guide schedule pending clarification.

## Freeze gate

Before any final test call, record:

- Git commit or immutable source archive hash;
- `student.py` SHA-256;
- config SHA-256;
- checkpoint SHA-256;
- tokenizer and evaluator SHA-256;
- seed and processed target count;
- complete training/search cost;
- CPU FP32 validation timing and peak RAM;
- uncompressed inference-asset size;
- selected interpolation or averaging settings, if any.

After this record is created, no model-affecting change may be made in response
to test output.

## Final evaluation and reporting

After freezing, evaluate both the official baseline and the single frozen
submission checkpoint on the full test split. Repeated runs are permitted only
for timing/reproduction of the same frozen predictor.

The report must distinguish observed measurements from design hypotheses and
include the initial baseline, equal-target comparison, key ablation, final result,
compute cost, limitations, and failure analysis. It must stay within ten pages,
including figures, tables, and references.
