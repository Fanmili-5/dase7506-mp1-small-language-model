# Stage28: cheap checkpoint-average and compatible weight-soup screen

Stage22 used a fixed last-five average, but its periodic validation curve was
still improving at update 7,200. Before spending another full training run, this
stage tests whether the averaging window left quality on the table.

The predeclared candidates average the same Stage22 trajectory over its last
2, 3, 4 or 5 saved checkpoints. Three additional candidates interpolate the
deployed Stage22 last-five weights with compatible Stage18 H weights at Stage22
fractions 0.50, 0.75 and 0.875. Both sources have the same seed and exact deployed
prefix-copy architecture; this does not imply that interpolation must help.

All candidates have identical inference cost and add no gradient targets. CUDA
FP32 validation screens the seven fixed candidates, followed by one independent
CPU FP32 verification of the winner. No test scoring. A winner must then be
re-mixed with train-only counts and independently resource-qualified; otherwise
the current Stage27/Stage24 chain remains unaffected.
