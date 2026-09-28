# Stage219: test the Stage155 training-budget confound

## Reason and fixed decision

The protected Stage143 predictor has 1.399686162 complete-validation BPB,
while Stage155's last-five checkpoint average has 1.409877270. That comparison
does not isolate architecture: the Stage143 neural lineage records at least
255,225,110 primary training-target presentations, whereas the fresh Stage155
run presented 58,982,400. Stage155's validation curve was still decreasing
at 7,200 steps (1.419101 at step 6,000; 1.413183 at step 7,200 before
averaging). This supports one continuation experiment, not a claim that
Stage155 will reach 1.35 or satisfy the eventual CPU budget.

Continue the SHA-pinned Stage155 five-checkpoint training average
`c2fe32ba15b0f13b11b43247f058971dac5717ac37fd2acda1bbaa173277bce1`
with the already established Stage56 low-learning-rate recipe: 4,800 new
updates, batch 32, 256 tokens, fixed seed-17 sampling stream advanced past
the original 7,200 steps, fixed dropout seed 54017, fresh AdamW at peak
2e-4 with 50-step warmup and cosine decay to 2e-5. This presents 39,321,600
additional primary targets, for a nominal total of 98,304,000. Only supplied
train text enters gradients; the loader verifies and reads train and
validation, never test. Keep the Stage155 architecture/configuration fixed.

Score the complete validation split every 300 updates. Before seeing the
results, select the lower of the step-4,800 endpoint and the uniform
same-trajectory average of steps 3,600/3,900/4,200/4,500/4,800. Compare
against Stage155's 1.409877270 reference on all 376,599 validation targets.
If selected BPB fails to improve by at least 0.008, stop this path. If it
passes, inspect whether it also beats Stage143; do not call it deployable
until trained-checkpoint CPU FP32 validation and all three resource limits
pass. A further continuation, if justified by the curve, needs a new
pre-outcome plan. No test access is authorized by this experiment.

The strongest objection is that extra updates may overfit the small corpus
or remain too slow on CPU. The fixed validation curve can reject the first;
full trained-model resource qualification is required for the second. The
student's acceptance levels remain <1.35 validation and <1.38 test; merely
passing this continuation gate is not completion.
