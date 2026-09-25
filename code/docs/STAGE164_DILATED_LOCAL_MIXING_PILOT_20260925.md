# Stage164: multiscale causal-convolution receptive field

## Frozen hypothesis and screen (before quality results)

Stage143 scores 1.399686162 complete-validation BPB, still 0.049686 above
the desired 1.35. Width, depth, six-attention allocation, shared branches,
spelling residuals, frequency emphasis and compact retrieval failed their
respective gates. Stage54's four local mixers each see only seven input
positions, while its alternating attention layers provide global context.
The untested question is whether a *hierarchy of local phrase spans* helps
the medium-frequency, out-of-prefix targets identified in Stage151.

Change only the four Stage54 gated depthwise-convolution dilations at
one-based blocks 2/4/6/8 from 1/1/1/1 to **1/2/4/8**, keeping seven learned
taps per block. Left padding is six times each dilation, preserving output
length and strict causality. This adds no weights, width, heads, blocks,
losses, data or inference-time state. It may improve longer local patterns,
but the existing global attention could make it redundant, and dilated taps
can miss contiguous phrases. The last objection is why the first block stays
undilated.

Use seed 17, batch 32, the *same initial parameter values and sampled-window
sequence* as Stage54, the first 2,400 updates of Stage54's 7,200-update
learning-rate trajectory, its R-Drop/auxiliary objectives and only supplied
train text. Thus both runs present **19,660,800 primary next-token targets**
at the fixed comparison point. Before training, test equal initial state,
exact equivalence at all dilations 1, causal future-token invariance,
independent batch rows, normalized finite output and finite gradients. Run
the unmodified full-validation scorer every 300 updates and use the
predeclared step-2,400 endpoint for the decision; do not select an earlier
checkpoint after seeing the curve.

The control is Stage54 seed-17 step-2,400 complete GPU FP32 validation BPB
**1.519950369**. Continue to a full 7,200-step run *only* if Stage164's
step-2,400 BPB is at most **1.499950369** (gain >=0.020 BPB), without
changing the dilation pattern. This gate reflects the remaining 0.049686
gap and the cost of training/export. A passing pilot is not a submission
candidate: final promotion would require the predetermined last-five
checkpoint average, same train-only count/gate pipeline or a separately
predeclared new one, complete CPU FP32 validation below Stage143, exact
compact OpenVINO export, and three alternating CPU/RAM/asset runs within
5x/4GiB/64MiB plus clean-extract reproduction. The causal and normalized
output contract is mandatory. Test remains untouched until method freeze.

The existing Stage143 checkpoint and ONNX graph are protected. Pilot outputs
go to new, non-overwriting Windows run/log directories. A failed quality or
correctness gate stops this route without resource qualification or test.
