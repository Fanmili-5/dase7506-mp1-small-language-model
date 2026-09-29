# Stage217: query-conditioned gate after global softmax attention

The current qualified Stage143 is 1.399686162 BPB on complete validation,
0.049686162 above the student's <1.35 aim. The Stage214 complementarity
audit found distributed contextual errors, while Stage216's learned static
relative-distance bias gained only 0.002174 BPB in its matched pilot. A
distinct hypothesis is that each token should suppress or amplify individual
global attention heads *after* content retrieval, rather than adjust the
attention weights by distance. This is inspired by the head-specific
post-SDPA gate in Qiu et al., **Gated Attention for Large Language Models**
([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)). Their large-scale
result does not predict a gain on this small WikiText-2 assignment.

Only the four global-attention blocks in Stage54's alternating hybrid gain
a per-head linear gate from the **current normalized hidden state**. The
head output is multiplied by `2 * sigmoid(gate)`. Gate weight and bias start
at zero, so the initial multiplier is exactly one and the model initially
equals Stage54. The factor two is our exact-start adaptation, not a claim to
reproduce the cited paper literally. The four convolution blocks, RoPE,
width/depth, prefix-copy head, R-Drop and auxiliary losses, sampler, seed,
fixed tokenizer and supplied train/validation split remain unchanged.

Predeclared sequence:

1. Test exact zero-start FP32 log-probability parity <=1e-5, normalized
   outputs <=1e-5, prefix causality <=3e-5, row independence <=1e-5 and
   finite gate gradients. These are correctness checks only.
2. On Windows, export a random-weight FP32 OpenVINO feature graph. Require
   hidden parity <=3e-4, eight-pair candidate/Stage143 feature-time ratio
   <=1.30, and conservative Stage143 asset projection <=64 MiB. Require two
   BF16 batch-32 synthetic CUDA updates with finite gradients and peak GPU
   reservation <8 GiB. This is **not** trained-predictor resource approval.
3. If preflight passes, run one fixed seed-17, batch-32, 2,400-update pilot
   with Stage54's first 2,400 of 7,200 learning rates and identical primary
   target count (19,660,800). Score the complete validation split every 300
   steps; at the **fixed endpoint**, require BPB <=1.489950368612022, a
   gain >=0.030 against the matched Stage54 control 1.519950368612022.
   Earlier points cannot be selected instead. Do not search gate bias,
   topology, seeds, learning rates or mixture weights after seeing results.
4. Only a passing pilot permits one fresh 7,200-step full run and fixed
   last-five checkpoint average. Promotion additionally requires complete
   CPU FP32 validation below Stage143 and three fresh full-predictor CPU,
   RAM and asset measurements under 5x/4 GiB/64 MiB. Test stays unopened
   until a replacement method is frozen. Stage143 is protected.

Hash-pin all sources/config/data and record the Windows task status,
checkpoint hash, validation trajectory and stop decision.
