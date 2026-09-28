# Stage221: shared-weight extra attention/conv pair

## Distinct architecture hypothesis

Stage143 remains at 1.399686162 complete validation BPB, short of the
student's <1.35 aim. Plain ten-block depth gained only 0.011507 at the
matched 2,400-step point (Stage150), while larger width+depth Stage155
improved neural-only quality but still did not beat Stage143 after a
completed full run and a fixed continuation. Stage219's extra 4,800 steps
gained only 0.003748. A remaining structural question is whether more
iterative global/local computation can help *without* giving the small
training corpus more independent parameters to overfit.

Stage221 keeps Stage54's 288-wide, eight-head, R-Drop Transformer and its
input/output/copy/auxiliary objectives, but executes ten logical blocks.
Logical blocks 1–8 are unique and alternate global attention with gated
causal convolution. Logical block 9 calls the exact module of block 7;
logical block 10 calls the exact module of block 8, including their norms
and FFNs. This is genuine parameter sharing, not a second independent
expert or an extra inference asset. The ten-block residual initialization
is fixed at construction. No data, tokenizer or evaluator changes.

Before any quality result, require structural checks of object identity,
exact training/inference export, finite normalized predictions, causal
prefix and independent-row behavior, and nonzero finite gradients. On
Windows, use a random-weight FP32 OpenVINO feature graph for a strict
input-only screen: eager/compiled maximum hidden error <=3e-4; graph plus
25,000,000 bytes of compact head/count reserve plus 200,000 source bytes
<=64 MiB; eight interleaved 32x256 four-thread feature calls at most
1.25x the Stage143 graph. A synthetic BF16 batch-32 two-update test must
fit the 8-GiB RTX 3070 Ti and have finite loss/gradients. These gates are
feasibility only, not final resource qualification.

If all pass, train one seed-17 2,400-step pilot using Stage54's batch-32
sample stream, unchanged R-Drop objective and the **first 2,400 learning
rates of its 7,200-step schedule**. Score complete validation every 300
steps. Compare the fixed endpoint with Stage54's same-step 1.519950369 BPB
over 376,599 targets. Require a gain of at least **0.030 BPB** to justify
a separately planned full 7,200-step run; do not choose an intermediate
checkpoint or adjust the sharing pattern after reading results. A passing
pilot would still need full CPU FP32 score and exact <=5x CPU, <=4-GiB
RAM, <=64-MiB asset qualification. Test remains untouched until a new
method freeze. The protected Stage143 bundle is never overwritten.

Strongest objection: extra computation may cost CPU but merely repeat the
same features, and sharing norms may restrict optimization. The matched
pilot and resource screen can reject this exact hypothesis. Passing them
would not prove <1.35; failing either stops the route.
