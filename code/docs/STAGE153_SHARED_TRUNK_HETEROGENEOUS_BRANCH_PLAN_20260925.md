# Stage153: shared-trunk heterogeneous two-path model

Stage91's two independently trained neural models combined to 1.3832024074
validation BPB, but their full ensemble exceeded the assignment's CPU and
asset limits and still did not reach 1.35. Stage147/150 show that simply
increasing width or serial depth gives only about 0.0115 BPB at the matched
2,400-step pilot point. Stage153 asks a different, resource-constrained
question: can two *heterogeneous upper paths* preserve useful ensemble
diversity while sharing a six-block trunk, token embeddings and output head?

The frozen shape is Stage54's width-288 eight-block backbone through block 6.
Path A retains attention block 7 followed by convolution block 8. Path B
adds convolution block 7 followed by attention block 8. Both read exactly the
same causal trunk state and use separate final RMS norms, but share the tied
vocabulary head, prefix-copy projections and gate. Their individually
normalized log-probabilities are averaged at fixed weight 0.5. At training,
the primary objective is the likelihood of that **normalized mixture**, not
an unnormalized logits average; shared deep supervision remains at layers
4/6, and both branch endpoints receive the fixed future-token auxiliary
loss. R-Drop uses the mixture distribution. Only the supplied training text
may set parameters. No test or cross-window state is allowed.

First verify causal, independent-window, normalized finite outputs and finite
training gradients on a small synthetic input. Then export a random-weight
two-hidden-state OpenVINO graph and compare it to Stage143 on validation
**inputs only**. Require max hidden error <=3e-4, graph bytes plus the same
25 MB compact head/count reserve and 0.2 MB source reserve <=64 MiB, and
eight interleaved four-thread CPU FP32 feature calls <=1.25x Stage143 median.
This is only a feasibility gate; it does not establish complete predictor
scoring resources or quality.

If it passes, a seed-17, 2,400-step matched pilot may train with the same
batch/window sampler and first 2,400 updates of Stage54's 7,200-step
learning-rate schedule, with a complete GPU FP32 validation score at every
300 steps. Compare step 2,400 to Stage54's 1.519950369 BPB, requiring at
least **0.020 BPB** gain to justify a full 7,200-step run. The higher gate
reflects extra implementation/resource cost and the remaining 0.049686 gap
to 1.35. Passing the pilot still requires a fresh train-only MKN mixture,
exact output export, three-repeat CPU/RAM/asset qualification, and clean
reproduction before promotion. The current Stage143 checkpoint is protected.

Strongest objection: shared lower layers and output head may remove the
diversity that made the over-budget ensemble work, while dual copy heads
could make CPU time too slow. The structure and resource preflight reject
these possibilities before spending GPU time; no improvement is assumed.
