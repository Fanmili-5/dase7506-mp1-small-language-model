# Stage169: complementary-teacher transfer into the stronger deployable student

This is a validation-development experiment, not a submission or a claim of
novelty. The accountable student must understand and disclose AI-assisted
design and implementation. Only the supplied WikiText-2 train text is used
for optimization. The tokenizer, validation scorer, test split and Stage143
fallback remain unchanged.

## Observation, question and rivals

The resource-qualified Stage143 candidate scores 1.399686162 complete
validation BPB. A non-deployable 50:50 Stage105/Stage155 teacher mixture
scores 1.376182664 on validation. Distilling that teacher into the Stage155
student reached 1.403105391 after a bounded continuation; this does not
establish that a Stage143-capacity student cannot transfer the teacher.

Question: can the same fixed teacher improve the *existing stronger*
Stage103/105-equivalent student under the final single-model inference
budget? Hypothesis: initialization and the integrated order-six MKN/copy gate
make the student easier to improve than Stage155 alone. Rivals: the apparent
complementarity may be unrepresentable in the smaller neural backbone; hard
train-label fine-tuning may cause the same gain without teacher information;
distillation may damage a useful pre-existing count/neural calibration.

## Frozen feasibility gate, before pilot outcomes

Load the SHA-pinned Stage105 and Stage155 checkpoints. Make the fixed
Stage105/Stage155 50:50 probability teacher. Initialize a differentiable
Stage103 student from the *exact Stage105 state*, without changing its
count tables, gate calibration or architecture. On one train-only batch of
eight independent 256-token prefixes:

1. Stage103 and Stage105 target log probabilities must agree within 3e-4,
   and the teacher must normalize within 1e-3.
2. A 0.75 teacher cross-entropy + 0.25 hard-label NLL backward pass must
   have finite loss and gradients, peak allocated CUDA memory <=6.5 GB and
   one-step post-initialization wall time <=6 seconds.

Any failure stops this route before a quality pilot. This checks only one
batch, not full-model validation, inference portability or resource limits.

## Fixed quality pilot, only if feasible

Two fresh, independently initialized students start from the identical
Stage105 state. Both use seed 169017, the same sampled train windows, physical
and effective batch eight, 900 updates, AdamW (betas .9/.999, decay .1),
peak LR 2e-5, 50-step warmup and cosine decay to 0.1 peak. The **teacher
arm** uses 0.75 full-distribution teacher CE + 0.25 hard NLL. The **control
arm** uses hard NLL only; it tests whether any improvement is merely further
training. The teacher is fixed and uses train prefixes only. Each arm presents
1,843,200 primary next-token targets, from the same sampling stream.

Score the unchanged full validation split (376,599 targets) at steps
0/300/600/900; the *fixed endpoint* is the decision point. Require teacher
endpoint at least 0.005 BPB below the 1.399686162 Stage143 benchmark **and**
at least 0.003 BPB below the matched hard-only control to license a separately
planned continuation. Report the entire curve, not a selected intermediate
checkpoint. If either gate fails, stop. The development target remains <1.35,
but neither this pilot nor its hypothetical success proves it.

Even a passing pilot must be exported into the compact Stage143 OpenVINO
single-copy form, pass exact parity and complete CPU FP32 validation, then
three fresh-process CPU/RAM/assets measurements (<=5x, <=4 GiB, <=64 MiB)
and clean-extract reproduction before replacing Stage143. No test scoring
until a subsequent explicit method freeze. The same validation split has
informed prior work, so this is exploratory development with a predeclared
internal comparison, not independent confirmation.

## Feasibility result, before quality pilot

The exact Windows 3070 Ti train-only preflight passed: maximum Stage103 versus
Stage105 log-probability difference was **0.0001659393**, fixed-teacher
normalization error **8.624e-7**, and all 59 inspected gradient tensors were
finite and nonzero. One backward step used **935,003,648 peak allocated CUDA
bytes** and **0.50018 seconds** after initialization. The SHA-pinned raw
record is `../results/stage169-evidence/preflight.json` (canonical JSON SHA-256
`81b1cd227c630e55c1cacca97741b70e302c6ded9ea331b3b584b0de7db7787e`,
so Windows/Mac line endings do not change the identity). This authorizes the
two-arm pilot under the fixed rules above; it establishes no validation gain.

## Matched pilot outcome and stop decision

The scheduled Windows task completed successfully. Both arms started at
**1.399686190 BPB** on the complete 376,599-target, 1,148,007-byte
validation split, agreeing to 1.3e-11 BPB. Each ran all 900 updates and
presented **1,843,200** primary next-token targets from the same seeded
training-window stream. The teacher arm's predeclared 300/600/900-step BPBs
were **1.399783118 / 1.399510987 / 1.399440139**; the hard-only arm's were
**1.403014529 / 1.403876749 / 1.403915928**. The teacher arm took 211.87
training seconds versus 137.40 for hard-only, excluding 40.11/39.44 seconds
of validation respectively.

At the fixed endpoint, teacher transfer beats Stage143 by only **0.000246023
BPB**, below the prespecified **0.005** continuation requirement. It beats
the matched hard-only arm by **0.004475789 BPB**, above the second 0.003
requirement, so the distinction is informative: additional hard-label
training damaged validation while teacher targets largely protected it, but
did **not** transfer enough quality to justify another long run. The
conjunctive gate fails. Neither the teacher nor control pilot is promoted,
exported to OpenVINO, resource-qualified or test-scored. Stage143 remains
the qualified development candidate at 1.399686162 BPB.

All four validation measurements per arm used the complete unchanged split.
The 17 source hashes in the Windows run record match the tracked sources;
the task ended with success and the fixed-file verifier passed both before
and after training. Raw plan, curves, metrics, console transcript and task
status are in `../results/stage169-evidence/`. The pilot checkpoints remain
on the Windows experiment host and are not inference or submission assets.
