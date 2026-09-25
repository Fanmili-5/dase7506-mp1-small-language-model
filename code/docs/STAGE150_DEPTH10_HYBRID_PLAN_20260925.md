# Stage150: deeper global/local Transformer under the fixed resource envelope

The accepted Stage143 predictor remains at 1.399686162 validation BPB,
0.049686 above the desired 1.35. Stage147's width-320 early pilot gained only
0.011346 BPB and missed its continuation gate. Stage148/149 show no large
train-text lookup or pairwise-topic complement. The next falsifiable
architecture question is depth, not another width or count coefficient:
the historical Stage112 seven-block repair was distinctly worse than the
eight-block model, suggesting that additional iterative global/local mixing
may matter. That observation is not proof that ten blocks will help, and the
Stage146 train/validation gap warns that extra capacity might overfit.

Stage150 keeps Stage54's width 288, heads, tokenizer, input/output/copy heads,
R-Drop and auxiliary objectives. It adds two blocks to reach depth ten and
extends the original alternating attention/causal-conv pattern by one
attention block and one conv block (`conv_layers=[2,4,6,8,10]`). The middle
supervision layers remain `[4,6]`. No course-fixed file or test split changes.

First run an input-only Windows CPU FP32 OpenVINO preflight against the exact
Stage143 graph. A random-weight graph must match eager hidden features within
3e-4, and its bytes plus a conservative 25 MB compact checkpoint/count/head
reserve and 0.2 MB source reserve must fit 64 MiB. Its eight interleaved
batch-32 feature calls must have median latency no more than 1.25x Stage143.
Passing only licenses an early training pilot, not final qualification.

If the preflight passes, train 2,400 seed-17 updates with the unchanged
Stage54 batch/window sampler and **first 2,400 updates of its 7,200-step
learning-rate trajectory**, presenting 19,660,800 primary targets. Compare
complete GPU FP32 validation at step 2,400 with Stage54's archived
1.519950369 BPB. Require at least **0.015 BPB** improvement before paying for
a full 7,200-step run. Any full model would still need exact train-only count
mixture rebuilding and independent CPU time/RAM/asset qualification. The
development score comes from validation only; the test split is untouched.

Strongest objection: more depth raises inference cost and may just memorize
the small supplied training corpus. The paired resource preflight and matched
early validation trajectory are low-cost ways to reject that hypothesis;
neither can guarantee reaching 1.35 at the final checkpoint.

## Input-only preflight result

The random-weight ten-block graph has **9,766,945** neural parameters and
occupies **39,125,349 bytes**. Adding the predeclared 25 MB count/head
checkpoint reserve and 0.2 MB source reserve yields **64,325,349 bytes**,
below the 64 MiB limit of 67,108,864 bytes. Eight interleaved Windows CPU
FP32 feature timings had medians **1.628874 s** for the candidate and
**1.318449 s** for Stage143, ratio **1.235447x**. Maximum hidden error
against eager PyTorch was **3.94e-6**. All three fixed pilot gates pass.
This does not establish complete predictor runtime or trained quality.
Hashes and raw timings are in `../results/stage150-evidence/preflight.json`.
