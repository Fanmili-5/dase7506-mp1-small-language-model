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
