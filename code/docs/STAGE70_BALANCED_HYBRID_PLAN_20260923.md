# Stage70: balanced global/local architecture preflight

Stage54's width-288 model with four attention layers is the strongest trained
backbone, while Stage59's width-320 model with only two attention layers is
0.00103 BPB worse despite faster early convergence and more CPU headroom.  The
unmeasured middle point may retain enough global routing while reallocating
compute to a wider representation.

Stage70 therefore tests an 8x304 hybrid with attention in layers 1/4/7 and
gated kernel-7 causal convolution in layers 2/3/5/6/8.  All blocks use
SwiGLU684.  Prefix-copy64 and the fixed collapsed MKN expert at weight .075 are
included in the preflight graph.  This changes an architectural compute
allocation, not a seed or validation-selected training setting.

Before gradient training, the complete random-weight inference bundle must pass
three fresh-process measurements: CPU scoring at most 5x baseline, peak RSS at
most 4 GiB and uncompressed inference assets at most 64 MiB.  Causality,
normalization, gradients and training/export parity are tested first.  Passing
only permits one fixed matched-budget R-Drop run; it is not a quality claim.
Test remains untouched.

## Result

All six architecture, causality, gradient and export tests passed.  The complete
random-weight graph used 8,163,313 neural parameters, 49,448,048 conservative
asset bytes and 2,037,997,568 bytes peak RSS.  Its three-repeat median CPU ratio
was **5.027017406x**, narrowly above the hard 5x limit.  Stage70 therefore fails
the resource preflight and does not receive gradient training.  The 0.54%
timing excess makes it a candidate only after an independently verified
inference-kernel speedup; it is not a submission candidate.  Test was not
scored.
