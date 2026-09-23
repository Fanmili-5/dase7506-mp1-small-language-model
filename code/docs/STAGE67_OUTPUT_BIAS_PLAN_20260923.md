# Stage67: train-fitted vocabulary intercept

Stage65's byte composition is positive but only improves the neural model by
0.0007774 BPB. Stage67 changes mechanism: the tied embedding/output matrix must
represent both token features and class boundaries, while its vocabulary head
has no intercept. A 2,048-value output bias can correct systematic per-token
frequency residuals at negligible inference cost.

The bias starts at exact zero on the frozen Stage65 neural checkpoint, so the
initial predictor is identical. Only those 2,048 values are optimized, using
five deterministic full passes over the supplied training tokens split into the
same independent 256-token windows as evaluation. The backbone, copy head and
all other tensors are frozen; validation supplies no gradient. The fixed
candidate is the uniform parameter average of epochs 3/4/5, followed by an
independent CPU FP32 validation score.

This is not a validation-fitted calibration: all learned values see training
text only. A positive neural result must still be rescanned with the unchanged
train-only MKN expert and pass the full 5x CPU, 4 GiB RAM and 64 MiB asset gate.
No test split is scored.

## Result

The exact-zero starting score was 1.4153287693 BPB. Validation improved
monotonically over the five full training passes to 1.4150970384. The fixed
epoch-3/4/5 average independently scored **1.4150978608 BPB** on CPU FP32
(checkpoint SHA-256
`3be9468b122da4c486726e8bcc59692005d0fc90dcbdd6a59c41b24d42d20b0f`).
This is a **0.0002308859 BPB** gain over Stage65. The mechanism is positive but
too small to become the main search direction; it advances only to one fused
MKN scan/resource check because it adds just 2,048 inference parameters. No test
split was scored.
