# Stage199: reassess full parallel attention/convolution under the actual course budget

Stage177 was stopped by its predeclared **feature-only** pilot screen of
`<=1.25x` the Stage143 graph, not by a complete course resource test or a
quality result. It had 1.438569x feature time, 3.81e-6 FP32 hidden parity
error, and projected assets of 61,372,843 bytes (<64 MiB). No Stage177
training/validation score exists. Stage186's quarter-width parallel branch
gave only 0.005196 BPB at 2,400 steps; that is negative evidence for the
narrow version, not a direct measurement of Stage177's four full-width
branches.

The actual course limit is complete CPU scoring <=5x baseline. On the same
Windows host, Stage143 validation took a median 86.051106 s and baseline
23.786126 s. The measured Stage177-vs-Stage143 feature difference was
1.788710-1.243395 = 0.545315 s per 32-window call. There are 46 calls in
complete validation; adding 46*0.545315 to Stage143 gives **111.14 s**, or
**4.67x** baseline. This is an arithmetic feasibility projection only:
trained weights, count/gate integration, cache effects and process overhead
may shift the actual ratio. It does not reverse Stage177's recorded failure
of its own stricter screen; this is a new, explicitly course-budget-based
prospective decision.

Before seeing new validation quality, fix the Stage199 experiment:

1. Use the *unchanged* Stage177 eight-block width-288 architecture, with
   full-width parallel RoPE causal attention alongside convolution in blocks
   2/4/6/8. Preserve seed 17, Stage54's R-Drop objectives, batch 32,
   256-token context and the first 2,400 learning rates of its 7,200-step
   schedule. The matched Stage54 step-2400 control is 1.519950369 BPB.
2. First run two synthetic BF16 AdamW updates at physical batch 32 on the
   RTX 3070 Ti. Require finite loss/gradients, normalized causal outputs,
   zero future-prefix effect, and peak reserved VRAM below installed total.
   If the fixed physical batch fails, stop instead of silently changing
   effective batch or schedule. No data split is opened in this preflight.
3. If it passes, train one 2,400-update seed-17 pilot, presenting exactly
   19,660,800 primary next-token targets, from supplied train text only.
   Score complete validation every 300 steps; the **step-2400 endpoint** is
   the selection gate, not the best intermediate step. The loader must not
   open test text.
4. Require >=0.030 BPB endpoint gain over the matched Stage54 step-2400
   control before a separate full-run decision. Such an early gain is
   necessary screening evidence, not a forecast of <1.35 or resource
   compliance. If the gate fails, archive and stop without another width,
   head, seed or learning-rate search. If it passes, a new predeclared full
   run, train-only MKN construction, complete compact CPU FP32 validation,
   three fresh-process CPU/RAM/assets checks and clean-extract reproduction
   remain mandatory before considering method freeze. No test access now.

The strongest objection is that four added attention paths may overfit the
small corpus or push the *complete* predictor over 5x CPU even if the pilot
gains quality. The fixed pilot tests quality and the later formal resource
gate would test actual eligibility; the projection alone proves neither.
