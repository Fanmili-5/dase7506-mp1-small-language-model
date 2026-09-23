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
