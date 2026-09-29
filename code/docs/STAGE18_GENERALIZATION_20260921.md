# Stage18: targeted regularization, identical inference Transformer

Preregistered before training: two independent changes to Stage15 F, not a
parameter/seed sweep. No larger teacher or extra training duration in this stage.

Launch: `MP1-stage18-20260921-a`, 2026-09-21 15:06:48 UTC. H passed the two-update
smoke and resumed, observed at update400. I is queued. Local suite80 tests:
75 passed/five CUDA skipped. Windows seven model/export/resume tests and two
orchestration tests passed, including CUDA BF16 backward. Read-only export
parity checks also passed using the full-width/depth real F checkpoint for both
regularizer configs; these checks did not write candidate weights or scores.
Startup receipt and source hashes are under `results/stage18-launch*`.
The original launch observations above are historical. Both jobs subsequently
completed at 15:39 UTC on September21: H average CPU validation1.482379119,
I1.482562139. Each consumed58,982,400 gradient targets. Gains over F were
.002715179 and .002532158 respectively: neither triggered the original .003
final-resource retest. These are small single-seed validation improvements,
not a final submission or a statistical-significance result. Stage19 separately
measures H resources and tests train-only count complementarity; it does not
rewrite Stage18's gate. Completed receipts are collected with Stage19 work.

## Observations, not assumptions about model age

Stage17 teacher failed its fixed quality gate: CPU validation BPB 1.665982500.
Its best periodic validation was 1.521367639 at update4200; the endpoint was
1.679397928. Training loss kept falling as validation deteriorated. The fixed
last-five average was retained as the declared result, not replaced post hoc by
the best checkpoint. Neither CE nor KD student trained. Extra gradient cost:
117,964,800 targets; teacher training3640.74s (validation/IO overhead additional).
This rejects this **teacher recipe**, not distillation in general.

Existing FP32 diagnostics, dropout disabled, identical 1024 sampled training
windows (seed1729), complete validation; no weight updates:

| Model | Sampled train mean NLL | Validation mean NLL | Gap |
|---|---:|---:|---:|
| Stage17 teacher | 1.401083 | 3.520151 | 2.119068 |
| Stage15 F | 2.539657 | 3.137942 | .598285 |

The teacher curve and gap jointly support overfitting. F's smaller gap alone
does not prove regularization will improve F: the following are hypotheses.
Training sample metrics are in-sample NLL, not full-training BPB. Diagnostic
CUDA validation values are not substituted for CPU ranking measurements.

## Two fixed candidates

- H: input embedding **row** dropout .10. One Bernoulli mask per vocabulary row
  per training call, scaled by1/(1-p). Repeated tokens share their row mask.
  The output vocabulary projection is NOT masked despite tied weights; no
  in-place changes. Hypothesis: reduce reliance on fixed token embeddings while
  retaining an intact output distribution and causal copy branch.
- I: **SwiGLU hidden activation** dropout .20, after gate*value and before output
  projection. Hypothesis: reduce FFN feature co-adaptation. Existing .10 attention,
  embedding-element and residual-output dropout is unchanged in both candidates.
- No combined candidate, no dropout-rate grid, no additional loss or targets.
- Same initial weights, width256/depth8/heads8, RoPE/RMSNorm/SwiGLU/copy64 and
  6,855,169 trainable parameters as F. Same seed17, original trainer, batch32,
  7200 updates, AdamW/LR/cosine schedule. Extra stochastic masks necessarily
  change later dropout RNG; this is not a paired-randomness guarantee.

Each candidate:58,982,400 next-token targets, total maximum117,964,800. From
scratch, two-update smoke followed by exact-plan resume. Checkpoint average fixed
at6000/6300/6600/6900/7200. No choosing training duration after seeing the curve.

## Inference and decision gates

`student_regularized.py` only changes training computation. Export removes its
two config keys, sets the implementation back to unchanged `student_structured`,
and keeps every tensor unchanged. Full-context CPU FP32 export equality is
asserted before writing an inference checkpoint. Original sources remain pinned.
Export ancestry records the training checkpoint/source hash and probabilities;
regularizer code is a reproduction dependency, not an inference dependency.

Resource feasibility derives from the identical inference graph to qualified F,
so no separate untrained resource screen is needed. This does NOT waive the
final exact-weight resource gate: >=.003 lower full CPU validation BPB than F
triggers three fresh-process CPU/RAM comparisons plus total asset accounting.
F remains the control if neither candidate improves and passes. Its 4.878x CPU
ratio has little margin; no relaxation of the5x/4GiB/64MiB limits is permitted.

Unit tests cover equal initialization/RNG, exact zero-regularization control,
row-mask semantics without output-weight mutation, complete distributions,
causality/independence, export equality, gradients, bit-exact resume, CUDA BF16,
and bounded validation-only orchestration. Tests use synthetic IDs, never model
training data. No new test score, external data/weights or leaderboard claim.

Substantive AI assistance: analysis, implementation, tests and orchestration;
the user must understand/disclose this. This is a custom implementation using
PyTorch dropout, not a claimed reproduction of AWD-LSTM or another paper.
