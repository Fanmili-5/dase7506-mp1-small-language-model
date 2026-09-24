# Stage140: heterogeneous teacher into the balanced, faster backbone

Stage72/77 established that the 8x300, three-attention/five-convolution
backbone fits the complete neural+MKN graph at 4.712/4.858x CPU in separate
three-repeat audits. Stage92 distilled the Stage71/Stage76 heterogeneous
teacher into the *Stage71* backbone and reached 1.4017 BPB. The balanced
architecture has not been tried as that teacher's student. Its greater
width, lower FFN ratio and one fewer attention layer change inference
allocation while preserving one network per prediction.

Use exactly the frozen Stage71/Stage76 0.55/0.45 teacher, the supplied
training text, Stage25 order-five count expert, 0.75 teacher CE + 0.25 hard
NLL, batch 24, 1,800 updates, AdamW peak LR 1e-5, BF16 training and seed
92017 from Stage92. Initialize the student from the exact Stage76 averaged
checkpoint, add a zero-initialized vocabulary bias, and verify zero-bias
probability equivalence before training. No other architecture or seed
changes. Monitor full validation at step 0 and each 300 updates with the
frozen Stage86 calibration and count weight 0.075; prespecify the average
of 900/1200/1500/1800. Subsequently score that average with the fixed
Stage94 calibration and order-five count weight 0.0625. No validation labels
enter gradients or schedule selection.

Only if the fixed average reaches <1.4 BPB on complete validation will it
advance to a portable fused export and independent three-repeat CPU <=5x,
RSS <=4GiB and assets <=64MiB qualification. A good GPU score alone is
not a deployable result. Stage85 remains the qualified fallback. No test
scoring or submission during this experiment.

## Observed result

The Windows RTX 3070 Ti run completed all 1,800 fixed updates and
11,059,200 training-target presentations in 342.09 training seconds. The
zero-bias Stage76 wrapper matched its source probabilities exactly on the
input-only smoke batch. Full Stage86-calibrated validation moved from
**1.420678740** at step zero through **1.411053027/1.409557887/
1.408793096/1.408353177/1.408189856/1.408116370** at steps
300/600/900/1200/1500/1800. The prespecified four-checkpoint average
(SHA-256 `79526ef74cab775844eba2d6a089e98a3ca0dd271b529782fa224cb6e2a3490d`)
scored **1.408340134 BPB** with the Stage86 mixture and **1.408932260 BPB**
with the fixed Stage94/order-five mixture, each over all 376,599 validation
targets. The latter is 0.007225 worse than the Stage92 primary-architecture
average under the same Stage94 mixture.

The fixed quality gate fails; this is a better *balanced* student than its
starting point under the frozen Stage86 calibration, but not a sub-1.4
candidate. No fused export or fresh CPU resource audit was attempted.
The earlier Stage72/77 resource measurements are architecture context, not
qualification of the new trained checkpoint. Evidence is under
`../results/stage140-evidence/`; the reused generic Stage86 average scorer
labels its own purpose as Stage92, but its recorded neural SHA identifies
the Stage140 checkpoint. No test scoring occurred.
