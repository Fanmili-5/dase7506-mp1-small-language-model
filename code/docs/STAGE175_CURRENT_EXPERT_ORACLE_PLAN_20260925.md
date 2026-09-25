# Stage175: current Stage143 expert-complementarity ceiling

## Question and boundary

Stage143's complete validation BPB is 1.399686162, 0.049686 above the
aspirational 1.35 target. Its input-only neural/count gate has received many
small refinements. The often-cited 1.361491 hindsight oracle belongs to an
**older** neural/count pair (Stage22), so it cannot answer whether a better
gate between Stage143's *current* experts could bridge this gap.

Before another GPU run or gate implementation, evaluate the existing frozen
Stage143 checkpoint once on the **complete validation split**. At each target,
recover the target probability assigned by its current neural expert and its
current order-six MKN expert from the exact pre/post sparse-add quantities.
The checkpoint, graph, gate, tokenizer, evaluator and count tables remain
unchanged. Never read test. Check that the reconstructed current mixture
reproduces its known complete-validation score and that every target has one
paired record. A target-aware oracle takes the larger of the two component
probabilities at each position. This uses the true next-token label, so it is
an intentionally unattainable *lower BPB bound* for any mixture of these
**fixed** experts, not a legal inference rule or a leaderboard result.

Decision, fixed before observing the result:

- If this oracle is **at least 1.35 BPB**, no input-only gate between the
  unchanged experts can meet the goal; stop new gate training and prioritize a
  stronger core predictor.
- If it is **below 1.35 BPB**, the route is not mathematically excluded, but
  its deployable improvement still has to beat 0.049686 BPB and pass the
  CPU/RAM/asset gates. Record the oracle gap; do not automatically launch a
  gate search from an unattainable number.

Strongest objection: the oracle uses answer information and may be far too
optimistic. The result is only a feasibility filter, never an estimated
deployable score. A second objection is cancellation when recovering tiny
count probabilities from FP32 sparse additions; the diagnostic checks the
complete current-mixture BPB, rejects materially negative recovered values,
and clips only sub-1e-6 roundoff before taking the maximum.

## Complete-validation result and decision

The Windows CPU FP32 four-thread diagnostic completed all **376,599** targets
and **1,148,007** raw bytes in 46 batches. The unchanged Stage143 prediction
reproduced **1.399686162042141 BPB**. Target-only reconstruction of the
neural/count mixture gave **1.399686161937438 BPB**, differing by less than
2.98e-8 in any target probability. The unattainable per-target oracle scored
**1.300186476133089 BPB**; at **107,323** targets the current count expert
assigned a higher probability to the correct answer than the neural expert.
The oracle has **0.099499686 BPB** of hindsight gain over the current gate.

The preregistered mathematical-exclusion gate does **not** fire: 1.30019 is
below 1.35. However, a legal gate would have to capture about **49.94%** of
the oracle's hindsight gain to reach 1.35. Earlier validation cross-fits with
causal confidence features gave only milliscale improvements for related
experts, and the Stage100 train-only fit suffered a neural/count calibration
mismatch. Those are cautionary evidence, not a proof about every possible
Stage143 gate. The next justified action is a bounded **CPU-only**
out-of-half diagnostic with richer *causal* features and an explicit >0.05-BPB
advance gate, followed by train-only fitting if it passes. Do not deploy
validation-fitted coefficients or score test. No GPU training or checkpoint
promotion follows from this oracle alone.

Raw aggregate evidence, source and asset hashes:
`../results/stage175-evidence/current-expert-oracle.json`.
