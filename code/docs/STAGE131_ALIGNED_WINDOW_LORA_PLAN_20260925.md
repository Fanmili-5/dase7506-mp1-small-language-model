# Stage131: evaluator-aligned-window LoRA control

Stage130 shows that Stage129's rank-8 backbone-LoRA average does not improve
the two-neural teacher cross-entropy even on the last 10% of train text when
that text is scored as fixed independent 256-token windows. Stage129 instead
optimized random 257-token crops, whose token position relative to a reset
context varies each time. This stage changes **only** the training-window
sampler to the evaluator's independent, non-overlapping 256-token layout.
It is a controlled test of objective/evaluation geometry, not a new seed or
architecture search. The fixed-window train suffix is in-sample and does not
prove generalization by itself.

Start from the exact Stage92 averaged neural and the same frozen Stage71/76
teacher, count expert, Stage94 calibration, rank-8 LoRA over 32 backbone
linears, initialization seed **129017**, batch 24, 1,800 updates, AdamW peak
LR 0.002, weight decay 0.01, and 0.75 teacher CE + 0.25 hard NLL. Shuffle
the complete aligned training windows each epoch with the same seeded CPU
generator, cycling without dropping a full window. The last incomplete
training window is excluded, so all 11,059,200 presented training targets
are valid and the target budget exactly matches Stage129. Do not alter
teacher weights, tokenizer, data, validation split or inference graph.

Monitor full validation at 0/300/.../1800 and merge the fixed step
900/1200/1500/1800 checkpoints into ordinary FP32 networks, then average
those four merged checkpoints. Step 0 must reproduce 1.401708 BPB. If and
only if the average scores <1.4, proceed to exact fused export and three-run
CPU/RAM/asset qualification under the unchanged 5x/4GiB/64MiB gates.
Otherwise retain Stage85 and archive the negative result. Test remains
untouched throughout development.

The strongest objection is that random-crop exposure should already cover
all causal positions. If the aligned version still fails, the stage rules
out this specific geometry mismatch as a sufficient solution; it does not
rule out other objectives or architectures.
