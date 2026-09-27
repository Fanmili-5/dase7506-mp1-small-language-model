# Stage196 result: causal attention residual did not clear quality gate

Under the prospectively revised course-budget question, the unchanged
Stage194 attention residual was trained from the same frozen Stage105 base
in two matched train/validation-only arms. The unchanged baseline at step
zero was approximately 1.399686194 BPB over 376,599 validation targets and
1,148,007 UTF-8 bytes.

The 1,200-step teacher arm ended at its best **1.398657565 BPB**, a gain of
**0.001028597** from the Stage143 validation score. The hard-only arm never
beat initialization and ended at **1.401366065 BPB**. The predeclared
minimum 0.015-BPB improvement was not reached; the teacher-effect gate also
failed. The extra causal attention did not materially outperform Stage195's
attention-free residual (1.398492575 BPB). This architecture is stopped and
receives no formal resource qualification or test scoring. Raw matched-arm
evidence is `code/results/stage196-attention-residual-pilot-v1.json`.

This is evidence against these two fixed frozen-base residual recipes, not
a proof that all new Transformer architectures fail. It supports moving
attention from incremental frozen-feature patches to a stronger core
predictor or the deadline-critical submission decision. The protected
Stage143 inference checkpoint remains unchanged.
