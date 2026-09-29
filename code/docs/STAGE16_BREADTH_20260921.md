# Stage 16 — breadth before further local optimization

## Status update: Transformer mainline retained

The user subsequently requested continued Transformer optimization. LSTM and
gated-convolution implementations are **untrained prototypes**, retained for
traceability; their gradient training and resource screens are paused. They
are not measured failures, nor submission candidates.

The already-started train-count/Transformer mixture screen completed on Mac
CPU FP32. Fixed grid weights 0/.05/.10/.20/.35 scored respectively
1.499334119 / 1.489530441 / 1.487889082 / 1.492161629 / 1.509973546 validation BPB.
The unchanged official evaluator independently reproduced 1.487889082 at weight
.10. Assets total 31,839,381 bytes. CPU ratio/peak-RAM qualification remains
pending, so this is a reserve experiment, not a qualified replacement. No test
scoring occurred. Raw evidence is under `results/stage16-evidence/`.

Stage17 pursues train-only Transformer-teacher distillation; see its separate
preregistered plan. The historical design rationale below is preserved.

This stage implements the user's correction: the search space is constrained
causal predictors, not a predetermined Transformer family. B is a reference,
not an assumption about the winning family. No new Transformer variant or
gradient-training job is authorized by this stage's feasibility script.

## Constraint-derived candidate map

All predictors: train-only learning, unchanged BPE-2048/evaluator/data,
independent causal context <=256, complete finite normalized 2048-way outputs,
CPU FP32 <=5x baseline, peak process RSS <=4GiB, uncompressed assets <=64MiB.
The official README explicitly permits compact train-derived assets reused
across windows; evaluation-prefix state cannot persist. Neural components must
pass genuine gradient contracts. No dummy parameters are added to a lookup model.

The candidate pool includes current attention, LSTM, GRU, gated convolution,
depthwise convolution, compact absolute-discount counts, Kneser-Ney counts,
neural/count mixtures, train-only teacher distillation, ensembles, low-rank
layers and state-space models. The first screen chooses LSTM, gated convolution
and a pruned count/neural mixture because each changes the computational model
and has an independently testable cost hypothesis. This is not evidence that
unselected families are inferior. Distillation is a cross-family training
strategy, deferred until a competitive compact student/teacher exists.

## Fixed feasibility candidates

- LSTM: embedding/output width 256, recurrent width 448, three layers, tied
  vocabulary projection, dropout 0.1, zero hidden/cell state for every call.
  FP32 head. It is a custom regularized LSTM baseline, not AWD-LSTM reproduction.
- Causal gated convolution: width 256, ten residual GLU blocks, kernel 3,
  dilations [1,2,4,8,16,32,64,1,2,4], tied vocabulary projection, LayerNorm,
  dropout 0.1. Nominal receptive field 269 covers the allowed 256 input positions,
  but no prediction can see beyond its available prefix. Not a GCNN reproduction.
- Counts: orders 2..5, retain n-grams occurring >=3 times, absolute discount 0.75,
  additive unigram smoothing 0.1. Denominators use original unpruned context
  totals; discarded mass backs off. This is NOT modified Kneser-Ney. Fixed train
  statistics stored as CSR buffers, packed context keys use <=44 bits. The
  builder observes 3,613,343 train tokens; four counting orders account separately
  for 14,453,362 n-gram events, not fictitious gradient-training targets.
- Mixture: the fixed Stage-14 B average plus count model. One predeclared grid:
  count weights [0,.05,.10,.20,.35]. No fitting on test, no token-specific weights
  learned from validation answers. Standalone counts are a diagnostic, not a
  claim of passing the trainable-model contract as a standalone submission.

Statistics initial build uses only train text and tokenizer verified by manifest.
The complete checkpoint is ~10.55 MB, larger than the earlier ~6.5MiB key/value
estimate because real tables also need CSR indices and backoff values.

## Order of operations and interpretation

1. Validate prefix causality, first-token/short-window behavior, normalized full
   output, independent batch/window states, reload, real neural gradients and
   pruned-mass accounting against a manually calculated synthetic example.
2. Score the fixed mixture grid on full CPU FP32 validation. Reuse probabilities
   inside a batch for efficiency; NEVER label grid-scan timing as inference time.
   Reconstruct the selected positive-weight checkpoint and score independently
   with the unchanged official evaluator. Include ALL source/statistical assets.
3. Measure random LSTM/convolution checkpoints and the selected mixture in fresh
   four-thread CPU processes, three repeats each, alternating baseline order.
   Mac checks are machine-specific feasibility evidence, not Windows or course
   reference-CPU guarantees. Do not run resource benchmarks concurrently on the
   same machine or compete with an active training job there.
4. Advance only feasible candidates to a separately fixed, short multi-rung
   gradient-training plan; do not infer quality from untrained models. That
   training plan is not launched automatically in this stage.
5. Report every rejected route and actual cost. If no mixture improves, retain B.
   A positive validation gain is not a leaderboard/test claim, and needs final
   exact-checkpoint resources and clean-package verification before submission.

Stage 15 continues only its already-started bounded plan; this stage does not
restart, extend or silently stop it. Its partial results remain separate.

## Sources and AI disclosure

- Official package GUIDE.md and code/README.md: authoritative task constraints.
- https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html : recurrent
  interface; omitted states initialize to zero, no bidirectional recurrence.
- https://arxiv.org/abs/1612.08083 : gated-convolution language model background.
- https://aclanthology.org/W11-2123/ : efficient statistical LM query background;
  this custom full-distribution CSR implementation does not copy KenLM code or
  claim its latency. No external datasets or weights.

AI provided substantive constraint analysis, candidate selection, implementation,
tests and experiment orchestration. The user must understand and acknowledge this.
