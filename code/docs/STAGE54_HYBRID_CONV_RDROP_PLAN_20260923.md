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
