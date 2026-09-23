# Stage59: train the resource-admitted local-heavy architecture

Stage58 admits the full width-320, six-convolution/two-attention inference graph
at 4.7406x baseline CPU time.  Stage59 now runs the same seed-17 R-Drop protocol
as Stage54: batch 32, 7,200 updates, AdamW, identical learning-rate schedule,
row dropout, deep supervision, +2/+3 future prediction and R-Drop alpha 0.5.

Only the backbone allocation changes.  Selection is the prespecified average
of updates 6,000/6,300/6,600/6,900/7,200 followed by exact export and independent
CPU FP32 validation.  The neural average must beat Stage54's 1.4285940452 BPB
before any MKN mixture or continuation.  No test split is scored.
