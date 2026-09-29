# Stage54: train the resource-admitted hybrid with R-Drop

Stage49 proves that the 8x288 alternating four-attention/four-gated-causal-conv
inference graph fits all three resource limits. Stage47 independently shows that
R-Drop improves the accepted Transformer by 0.01438 BPB without changing its
deployed graph. Stage54 combines these established mechanisms and compares
directly against Stage47 under the same seed, sampled windows, 7,200 updates,
optimizer, schedule, two stochastic forwards, auxiliary objectives and fixed
five-checkpoint average.

Only the backbone changes. The hybrid replaces layers 2, 4, 6 and 8 with
kernel-7 gated causal depthwise-convolution blocks, widens the residual stream
to 288 and uses SwiGLU hidden width 720. R-Drop and auxiliary heads are removed
at export; exact output equivalence to `student_hybrid_conv_structured` is
tested before independent CPU FP32 validation.

The averaged neural model must beat Stage47 by at least 0.003 BPB before any
MKN mixture or full resource qualification. No test split is scored.

## Result

The Windows RTX 3070 Ti run completed all 7,200 updates and passed the fixed
file checks and export-equivalence tests.  The five-checkpoint average over
steps 6,000, 6,300, 6,600, 6,900 and 7,200 scored **1.4285940451894024 BPB**
under independent CPU FP32 validation.  This improves on Stage47's matched
R-Drop average (1.4506130031808826) by **0.0220189579914802 BPB**, well beyond
the 0.003 admission gate.  The exported neural checkpoint is 31,779,445 bytes
with SHA-256
`689101810f2165091647366ebaaa11e3c693edc9a80155a009237cc131a25d51`.

Stage49 already measured the identical deployed graph at a 4.880441205337089
CPU-time ratio and 2,039,488,512 peak RSS.  Stage54 therefore advances to a
fixed validation-only MKN weight scan and a fresh end-to-end resource audit.
The test split remains untouched.
