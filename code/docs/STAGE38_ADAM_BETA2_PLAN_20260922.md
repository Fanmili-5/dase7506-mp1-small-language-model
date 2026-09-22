# Stage38: modern AdamW second-moment timescale

All accepted neural runs inherited the baseline AdamW beta2=0.999 without a
matched check. Modern small-Transformer recipes commonly use a shorter
second-moment timescale. Stage38 changes exactly one training setting to
beta2=0.95 while restoring the accepted 8x256 Stage26 architecture and keeping
seed 17, peak learning rate 0.001, weight decay 0.1, schedule, batch 32, 7,200
updates, 58,982,400 primary targets, deep supervision and +2/+3 multi-token
prediction fixed.

The deployed graph is byte-for-byte the same architecture as Stage26, so this
is a training-only optimizer ablation and requires no random-weight resource
preflight. Selection remains the fixed average of updates
6,000/6,300/6,600/6,900/7,200 followed by independent CPU FP32 validation.
Only a quality gain will trigger renewed count mixing/calibration and exact
resource qualification. No seed grid, beta grid, adaptive duration or test
scoring.
