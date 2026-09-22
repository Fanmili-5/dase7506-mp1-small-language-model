# Stage35: 9x240 depth-reallocation preflight

After the 10x224 seven-head graph failed CPU timing, Stage35 tested one
resource-only follow-up: nine layers, width 240 and eight heads. Random seed-17
weights were serialized with the fixed collapsed modified-KN expert at weight
0.125. Three alternating fresh-process CPU FP32 comparisons were run before any
gradient training.

The median candidate-to-baseline ratio was **4.996884865x**, peak RSS was
2,047,414,272 bytes, and conservative inference assets were 43,780,632 bytes.
All formal thresholds pass, but the time margin is only about 0.062%, so this is
a training-admission result rather than a robust final qualification. The graph
contains 6,747,841 neural parameters, close to the current 8x256 model's size.

Stage36 is therefore allowed to train with the fixed matched Stage26 target
budget. Any quality-positive trained checkpoint must repeat the complete
resource measurement and will be rejected if its measured ratio exceeds 5x.
Stage35 used no gradient target and did not score test data.
