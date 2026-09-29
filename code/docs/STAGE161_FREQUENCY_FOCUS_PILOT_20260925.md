# Stage161: train-only medium-frequency unseen-prefix emphasis

The resource-qualified Stage143 model remains at 1.399686162 complete
validation BPB; the <1.35 target is not achieved. Stage151 found 57,247
validation targets with train frequency 100–999 and absent from the current
causal input prefix, averaging 4.700063 nats. The train/validation gap is
0.635818 nats per target on the recorded diagnostic. The static spelling
residual (Stage160) failed; exact retrieval and another Transformer branch
have also missed their advancement gates. Rather than add an inference-time
component, test whether training allocates too little gradient to this
well-defined error group. This is a training hypothesis, not a claim that
the group can be detected from its unknown answer at inference.

Use the unchanged Stage54 eight-block hybrid Transformer and its seed-17,
batch-32, R-Drop/auxiliary objective and 7,200-step LR curve. Change only
the main next-token loss during the first 2,400 updates: for each *training*
target, set `m=1` when its train-only unigram count is 100–999 and that
target ID has not appeared among the current input window's positions 0..t.
Replace the ordinary mean main loss by

`sum(nll * (1 + 0.25*m)) / sum(1 + 0.25*m)`.

Apply the same normalized substitution to both R-Drop passes. Leave the
deep/future losses, symmetric KL, optimizer, dropout, sampler, initialization,
batch, model, tokenizer, evaluator and data unchanged. The subgroup indicator
uses labels **only while computing the training loss**; at validation and
inference, the candidate's forward path is identical to Stage54. The
frequency table comes from supplied train tokens only and is not serialized
for inference. No validation/test labels enter training.

Preflight: verify same-seed model weights and eval predictions match Stage54
exactly at step zero, the group mask excludes a target seen earlier in its
window, and the normalized loss matches the ordinary loss when alpha=0.
Then score complete GPU FP32 validation every 300 steps. Compare only the
fixed step-2,400 endpoint against Stage54's archived same-seed/same-target
**1.5199503686120217 BPB**. A gain >=0.015 BPB is required to license
the fixed 7,200-step continuation and later Stage143 comparison. A positive
pilot does not itself prove any final or CPU-qualified gain. A smaller gain
or regression stops this route without inference export or test scoring.
The strongest objection is that upweighting target rarity can merely trade
away common-token accuracy; the full normalized validation BPB, not group
loss alone, is the decision gate.

## Completed pilot and decision

The same-seed preflight found exactly equal initial weights and predictions,
the expected toy causal mask `[true,false,true]`, and zero alpha-zero loss
error. The fixed Windows run completed 2,400 updates and 19,660,800 primary
training targets. The focus group occupied about 14.1–16.2% of sampled
positions in the logged batches. The complete GPU FP32 validation endpoint
was **1.521144191 BPB**, **0.001193823 worse** than Stage54's archived
1.519950369 matched endpoint. Gains at intermediate steps were small and
inconsistent; the largest was only 0.003788 BPB at step 600. This is far
below the predeclared >=0.015 gate. The route stops without full 7,200-step
training, inference export, CPU qualification, or test scoring. It does not
show that every data-dependent loss is ineffective, only that this specified
medium-frequency unseen-prefix weighting did not improve the matched pilot.
Evidence: `code/results/stage161-evidence/`.
