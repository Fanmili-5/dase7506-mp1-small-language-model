# Stage209 result: exact tree math repaired, PyTorch CPU gate missed

The Windows input-only preflight confirms exact equality with Stage208's
random-weight leaf log probabilities (maximum absolute error **0.0**) and
2,048-way normalization/future-position/row isolation within 4.77e-7.
One 32x256 BF16 R-Drop synthetic update had finite, nonzero lexical
gradients; peak allocated GPU memory was **5,812,288,000 bytes** and
projected inference assets **59,005,284 bytes**. No training or validation
text was opened, and no test score was produced.

Level-wise propagation lowered the eight-run median four-thread CPU
hierarchy-head time from Stage208's **3.103434 s** to **0.642773 s** per
32x256 batch, but this misses the predeclared **0.50 s** gate. Therefore
Stage209 itself is not admitted to the matched-target quality pilot. The
only remaining bounded engineering check is Stage210's separately
predeclared FP32 OpenVINO head screen; it does not change the math or relax
the quality gate. If that fails, stop the hierarchy route.

Raw measurements and hashes: [`../results/stage209-preflight-a.json`](../results/stage209-preflight-a.json).
