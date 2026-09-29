# Stage36: matched-target 9x240 result

Stage36 trained the resource-admitted 9x240/eight-head Transformer with the exact
Stage26 seed, 7,200 updates, batch 32, optimizer, learning-rate schedule,
58,982,400 primary targets, deep supervision and +2/+3 multi-token objective.
Only depth, width, head geometry and auxiliary layer indices changed.

The endpoint scored 1.4717289978 BPB. The prespecified average of updates
6,000/6,300/6,600/6,900/7,200 exported exactly to the unchanged structured
inference graph and scored **1.4685436410 BPB** on independent CPU FP32. This is
0.0035497327 worse than the matched Stage26 8x256 result, 1.4649939083. The
candidate is rejected without count mixing or final trained-weight resource
measurement. The experiment shows that one more block did not compensate for
the narrower representation under this fixed compute envelope. No test data was
scored.
