# Stage144: sub-1.35 quality-mechanism triage

The Stage143 local Windows candidate is reproducible at 1.399686162
validation BPB and 3.6177x baseline CPU time. The new target is below 1.35:
the remaining 0.049687 BPB is much larger than the gains from the recent
calibration, gate, copy, distillation and inference-backend changes. A faster
implementation alone cannot reach it. Keep Stage143 as the qualified fallback,
select only on validation, and leave test untouched until a new freeze.

The tension is strong sequence modeling versus a fixed 5x CPU / 64MiB asset
envelope. We separate quality mechanisms from speed/compression mechanisms;
each candidate must clear both. Raw candidates below are ideas, not scores.

| Candidate mechanism | Falsifying evidence or first gate |
| --- | --- |
| Exact train-text long-suffix retrieval | Measure causal within-window prefix coverage and correct continuations before building an index. It must add far more than the ~0.001 local-cache gain. |
| Approximate train-text semantic retrieval | Index and query cost/asset likely too high; only test if exact-match coverage is meaningful. |
| Wider/deeper single Transformer with OpenVINO | Stage143 has 1.38x time headroom and 11.3MB asset headroom; require a cheap architecture/resource preflight before full training. |
| Different attention/conv allocation | Stage76 is complementary but weaker alone; use equal-target and resource controls, not architecture age as a proxy. |
| Shared-parameter additional depth | Saves assets but not CPU; reject if feature latency approaches 5x before training. |
| Larger FFN ratio with fewer blocks | Stage112 showed a block matters; requires matched compute and full validation. |
| Joint backbone and successor-head training | Stage136 head-only repair failed; expensive, with a distinct failure explanation. |
| Neural + train-only word/character finite-state expert | Must show target-conditional gain beyond order-six MKN without future/context leakage. |
| Train-only topic-conditioned unigram prior | Static prior and gates yielded small gains; cheap diagnostic before training. |
| Compact heterogeneous second expert | Stage91 over-budget ensemble is only 1.38162, still not 1.35. Compression alone is insufficient. |
| New dropout/augmentation schedule | Continuations Stage86/109/120/125 regressed; require fresh controls, not another seed search. |
| Distill larger teacher into single student | Stage92/140 transfers were small; require teacher quality and transfer gate before spending another long run. |
| Better within-window pointer mechanism | Stage43/132 cache gain was ~0.001 and Stage136 successor route was worse; only revisit with new causal evidence. |
| Refit count/neural gate | Stage102 gain was milliscale; cannot plausibly close 0.05 alone. |

First bounded local pilot: scan only supplied train/validation token streams for
train-history matches at orders 6–32, respecting each independent 256-token
validation window. Record how often the actual next token was ever seen after
the same train history. This is a **hindsight diagnostic** using validation
labels for measurement, not an inference method, trained parameter, or BPB
result. If correct long-suffix coverage is too sparse to plausibly close a
material portion of 0.05 BPB, do not build a retrieval engine. Then prioritize
an architecture/optimization pilot on Windows once the tunnel is available.

Strongest objection: exact matches may overstate utility because the model
already predicts these tokens well and unmatched/wrong matches can hurt. A
coverage pass only licenses a next-stage *full probability* comparison, never
a claim of validation improvement or deployment readiness.

## Exact-prefix coverage result

The local diagnostic used all 3,613,343 supplied train tokens and only the
376,599 validation targets. At order seven, 18,099 of 367,767 eligible
within-window histories matched train (4.9213%); the true next token appeared
after that history for 11,271 positions (3.0647%). At order eight the same
figures were 11,217/366,295 (3.0623%) and 7,519 (2.0527%). A six-token
history had a 4.9250% true-successor rate. Stage143's order-six MKN reaches
five observed history tokens, so this particular six-token lookup is new;
however, its signal can still overlap the model's predictions. These are
coverage counts, **not** probability gains or an oracle BPB. Exact train
retrieval alone is unlikely to close 0.0497 BPB, but a full-probability pilot
is needed before ruling out a smaller complementary gain.
Full counts, source and data hashes: `../results/stage144-evidence/train-suffix-coverage.json`.

The next candidate is a fixed six-attention/two-convolution variant of the
eight-layer width-288 backbone. Stages54/59 suggest global/local allocation
matters; unlike seed rescreening, this changes token mixing. Stage143's
OpenVINO path offers CPU headroom that older PyTorch preflights lacked. First
compare random-weight FP32 feature graph parity, bytes and Windows CPU latency
against the accepted Stage143 graph; only then run a 2,400-update matched
R-Drop pilot. Retain the unchanged Stage54 2,400-step validation checkpoint
as the same-seed, same-target control. A pilot advantage of at least 0.015 BPB
is the gate for paying for a full 7,200-step trajectory; smaller advantages
cannot by themselves support a credible path from 1.3997 to 1.35. This gate
does not mean a 0.015 early advantage proves a 0.05 final gain.

The Stage145 input-only Windows preflight passed its fixed pilot gate. A
random-weight six-attention/two-convolution feature graph was 32,459,299
bytes, only 654,258 bytes larger than the Stage143 graph. Its OpenVINO FP32
four-thread feature median was 1.513403 seconds per 32x256 batch versus
1.299777 seconds for the Stage143 reference, a 1.16436x ratio below the
1.25x pilot cap. Maximum hidden-feature discrepancy from eager PyTorch was
3.58e-6. This is **not** a complete predictor resource result, a trained
model, or a BPB claim. Raw eight-pair timing and hashes are in
`../results/stage145-evidence/preflight.json`.

## Stage145 matched pilot outcome

The scheduled Windows job completed all **2,400** prespecified updates,
presenting **19,660,800** primary training targets in 693.55 training seconds.
The complete GPU FP32 validation result at step 2,400 was
**1.515251139 BPB** (376,599 targets), versus the unchanged same-seed,
same-budget Stage54 step-2,400 control at **1.519950369 BPB**. The measured
gain is **0.004699230 BPB**, below the predeclared 0.015 pilot gate. The
new allocation was initially worse, overtook the control after 1,200 steps,
and ended only modestly better. This is not evidence that full 7,200-step
training would reach or approach 1.35, so no long continuation is launched.
The pilot also shortened its learning-rate cycle to 2,400 steps whereas the
archived Stage54 control was still on its 7,200-step cycle at step 2,400;
therefore the 0.004699 margin is exploratory, not a strictly isolated
architecture effect. The next architecture pilot will preserve Stage54's
original learning-rate trajectory.
The full 300-step curve, source/checkpoint hashes, job status and transcript
are under `../results/stage145-evidence/`. No CPU qualification or test score
was attempted for this rejected exploratory model. Stage143 remains the only
sub-1.4 resource-qualified candidate.

Next, use the existing `diagnose_generalization.py` with the exact Stage143
checkpoint, four CPU threads, 1,024 sampled train windows (seed 1729) and
the complete validation stream. Compare train/validation NLL by within-window
position, whether the target was observed in the prefix, and train-token
frequency. This is an error-allocation diagnostic, not an optimization or
extra test. It will decide whether to prioritize data/regularization,
copy/count specialization, or further architecture work.

The observed Stage143 diagnostic reproduces 1.399686162 BPB. The 1,024
seed-1729 sampled *in-sample* training windows average 2.321660 nats/target,
versus 2.957478 on complete validation, a 0.635818 gap; it is not a paired
generalization estimate because the train sample is drawn from text used for
fitting. On validation, 228,403 targets absent from their current input prefix
average 3.587674 nats, versus 1.986206 for 148,196 seen targets. The
frequency-100–999 group also has a 4.316574-nat validation average. These
groups overlap and cannot be added. The data support investigating unseen-
context generalization, not a claim that more parameters alone will suffice.
Raw group sums and hashes: `../results/stage146-evidence/stage143-generalization.json`.

Before building an expensive suffix index, run a target-probability-only
diagnostic with the exact Stage143 predictor: save its probability for every
validation target under the fixed scorer, then use *only training text* to
form unsmoothed exact successor distributions for six-, seven- and eight-token
histories. For matched histories, try fixed mixture weights 0, .02, .05, .10
and .20; unmatched histories use Stage143 unchanged. Respect every independent
256-token window boundary. Report all cells, not only the best. Target labels
are used only for retrospective scoring, never in the lookup or deployment.
Require at least 0.01 BPB improvement on complete validation before attempting
a compressed index, causal implementation and official resource audit. A
positive target-only result is a ceiling-like diagnostic, not a valid
checkpoint or leaderboard result. No test scoring.

## Suffix probability result and decision

The cached Stage143 target log probabilities reproduced complete validation
at **1.399686162042141 BPB** with the checkpoint and source hashes intact.
The best cell in the fixed grid was six-token history, weight 0.10, at
**1.396648904 BPB**, a gain of **0.003037258**. Seven- and eight-token
histories reached only **1.396830820** and **1.396941995** at their best
fixed weights. All are well below the 0.01 BPB advancement gate. These
calculations used true validation targets only to read out probabilities and
measure loss; no target-dependent lookup is allowed at inference. Even the
diagnostic gain would explain only about 6% of the 0.049686 BPB gap to 1.35.
No suffix index or deployed model is built. The ignored target-probability
array can be regenerated from the exact Stage143 checkpoint; its SHA-256 and
the full score grid are recorded under `../results/stage146-evidence/`.
